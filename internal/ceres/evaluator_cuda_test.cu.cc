// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: jwmak@ucdavis.edu (Jason Mak)
//
// Test the CUDA-based evaluator using small bundle adjustment problems that contain
// cost functions of differing types, constant parameter blocks, loss functions, and manifolds.
// The residuals, gradients, and jacobians computed by the CUDA-based the evaluator
// should match those computed by the default host evaluator on the CPU.

#include "ceres/evaluator.h"

#include <memory>
#include <string>
#include <vector>

#include "ceres/autodiff_cost_function.h"
#include "ceres/internal/cuda_defs.h"
#include "ceres/internal/eigen.h"
#include "ceres/loss_function.h"
#include "ceres/loss_function_cuda.h"
#include "ceres/manifold.h"
#include "ceres/problem.h"
#include "ceres/problem_cuda.h"
#include "ceres/problem_impl.h"
#include "ceres/product_manifold.h"
#include "ceres/program.h"
#include "ceres/rotation.h"
#include "ceres/sparse_matrix.h"
#include "ceres/stringprintf.h"
#include "ceres/types.h"
#include "gtest/gtest.h"

namespace ceres::internal {

double const kTolerance = 1e-13;

struct EvaluatorTestOptions {
  EvaluatorTestOptions(LinearSolverType linear_solver_type,
                       int num_eliminate_blocks,
                       bool dynamic_sparsity = false)
      : linear_solver_type(linear_solver_type),
        num_eliminate_blocks(num_eliminate_blocks),
        dynamic_sparsity(dynamic_sparsity) {}

  LinearSolverType linear_solver_type;
  int num_eliminate_blocks;
  bool dynamic_sparsity;
};

static void SetSparseMatrixConstant(SparseMatrix* sparse_matrix, double value) {
  VectorRef(sparse_matrix->mutable_values(), sparse_matrix->num_nonzeros())
      .setConstant(value);
}

// A cost function that penalizes a point for moving from
// an initial starting position. This can be used to create
// and test a cost function with 3 residuals.
struct PointDisplacementError {
  PointDisplacementError(double x, double y, double z)
      : x_(x), y_(y), z_(z) {}

  template <typename T>
  HOST_DEVICE bool operator()(const T* const point,
                              T* residuals) const {
    residuals[0] = abs(x_) - abs(point[0]);
    residuals[1] = abs(y_) - abs(point[1]);
    residuals[2] = abs(z_) - abs(point[2]);

    return true;
  }

  static ceres::CostFunction* Create(const double x,
                                     const double y,
                                     const double z) {
    return (
        new ceres::AutoDiffCostFunction<PointDisplacementError, 3, 3>(
            new PointDisplacementError(x, y, z)));
  }

  double x_;
  double y_;
  double z_;
};

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionErrorNoRadialDistortion {
  SnavelyReprojectionErrorNoRadialDistortion(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  HOST_DEVICE bool operator()(const T* const camera,
                              const T* const point,
                              T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    AngleAxisRotatePoint(camera, point, p);

    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    const T xp = -p[0] / p[2];
    const T yp = -p[1] / p[2];

    // Compute final projected point position.
    const T& focal = camera[6];
    const T predicted_x = focal * xp;
    const T predicted_y = focal * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorNoRadialDistortion, 2, 7, 3>(
        new SnavelyReprojectionErrorNoRadialDistortion(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 10 parameters. 4 for rotation, 3 for
// translation, 1 for focal length and 2 for radial distortion. The
// principal point is not modeled (i.e. it is assumed be located at
// the image center).
struct SnavelyReprojectionErrorWithQuaternions {
  // (u, v): the position of the observation with respect to the image
  // center point.
  HOST_DEVICE SnavelyReprojectionErrorWithQuaternions(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  HOST_DEVICE bool operator()(const T* const camera,
                              const T* const point,
                              T* residuals) const {
    // camera[0,1,2,3] is are the rotation of the camera as a quaternion.
    //
    // We use QuaternionRotatePoint as it does not assume that the
    // quaternion is normalized, since one of the ways to run the
    // bundle adjuster is to let Ceres optimize all 4 quaternion
    // parameters without using a Quaternion manifold.
    T p[3];
    QuaternionRotatePoint(camera, point, p);

    p[0] += camera[4];
    p[1] += camera[5];
    p[2] += camera[6];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    const T xp = -p[0] / p[2];
    const T yp = -p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[8];
    const T& l2 = camera[9];

    const T r2 = xp * xp + yp * yp;
    const T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T& focal = camera[7];
    const T predicted_x = focal * distortion * xp;
    const T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return (
        new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorWithQuaternions,
                                        2,
                                        10,
                                        3>(
            new SnavelyReprojectionErrorWithQuaternions(observed_x,
                                                        observed_y)));
  }

  double observed_x;
  double observed_y;
};

double camera1[10] = {
 9.99946154126841180165e-01,
 7.87061670168454075025e-03,
 -6.39535329165887445751e-03,
 -2.20038540935716883662e-03,
 -3.4093839577186584e-02,
 -1.0751387104921525e-01,
 1.1202240291236032e+00,
 3.9975152639358436e+02,
 -3.1770643852803579e-07,
 5.8820490534594022e-13
};

double camera2[10] = {
 9.99877513605250900497e-01,
 7.98833588996764563939e-03,
 -1.26117173449355086945e-02,
 -4.69987892415464365153e-03,
 -8.5667661408224093e-03,
 -1.2188049069425422e-01,
 7.1901330750094605e-01,
 4.0201753385955931e+02,
 -3.7804765613385677e-07,
 9.3074311683844792e-13
};

double camera3[7] = {
 1.4846251175275622e-02,
 -2.1062899405576294e-02,
 -1.1669480098224182e-03,
 -2.4950970734443037e-02,
 -1.1398470545726247e-01,
 9.2166020737027976e-01,
 4.0040175368358570e+02
};

double point1[3] = {
 -6.1200015717226364e-01,
 5.7175904776028286e-01,
 -1.8470812764548823e+00
};

double point2[3] = {
  1.7074972220818254e+00,
  9.5386921723786655e-01,
  -6.8771685779735616e+00
};

static void EvaluateBundleAdjustmentProblem(SparseLinearAlgebraLibraryType sparse_linear_algebra_library_type) {
  Problem::Options problem_options;
  problem_options.cost_function_ownership = DO_NOT_TAKE_OWNERSHIP;
  problem_options.manifold_ownership = DO_NOT_TAKE_OWNERSHIP;

  Problem problem(problem_options);
  Manifold* camera_manifold = new ProductManifold<QuaternionManifold, EuclideanManifold<6>>{};
  CauchyLoss* cauchy_loss = new CauchyLoss(1.0);
  HuberLoss* huber_loss = new HuberLoss(1.0);

  CostFunction* cost_function1 =
      SnavelyReprojectionErrorWithQuaternions::Create(-3.326500e+02, 2.620900e+02);
  problem.AddResidualBlock(cost_function1, cauchy_loss, camera1, point1);

  CostFunction* cost_function2 =
      SnavelyReprojectionErrorWithQuaternions::Create(-1.997600e+02, 1.667000e+02);
  problem.AddResidualBlock(cost_function2, cauchy_loss, camera2, point1);

  CostFunction* cost_function3 =
      SnavelyReprojectionErrorWithQuaternions::Create(1.224100e+02, 6.554999e+01);
  problem.AddResidualBlock(cost_function3, cauchy_loss, camera1, point2);

  CostFunction* cost_function4 =
      SnavelyReprojectionErrorNoRadialDistortion::Create(-2.530600e+02, 2.022700e+02);
  problem.AddResidualBlock(cost_function4, huber_loss, camera3, point1);

  CostFunction* cost_function5 =
      PointDisplacementError::Create(point1[0], point1[1], point1[2]);
  problem.AddResidualBlock(cost_function5, nullptr, point1);

  CostFunction* cost_function6 =
      PointDisplacementError::Create(point2[0], point2[1], point2[2]);
  problem.AddResidualBlock(cost_function6, nullptr, point2);

  problem.SetParameterBlockConstant(camera2);
  problem.SetParameterBlockConstant(point2);
  problem.SetManifold(camera1, camera_manifold);

  ProblemImpl* problem_impl = problem.mutable_impl();
  Program* program = problem_impl->mutable_program();

  std::vector<double*> removed_parameter_blocks;
  double fixed_cost = -1;
  std::string error;
  std::unique_ptr<Program> reduced_program =
      program->CreateReducedProgram(&removed_parameter_blocks,
                                    &fixed_cost,
                                    &error);
  std::vector<double> state(reduced_program->NumParameters());
  reduced_program->ParameterBlocksToStateVector(state.data());

  Evaluator::Options options;
  // This will use BlockJacobianWriter to store the Jacobian.
  options.linear_solver_type = ITERATIVE_SCHUR;
  options.sparse_linear_algebra_library_type = sparse_linear_algebra_library_type;
  options.num_eliminate_blocks = 0;
  options.context = problem.context();

  std::unique_ptr<Evaluator> evaluator(
      Evaluator::Create(options, reduced_program.get(), &error));

  std::unique_ptr<SparseMatrix> jacobian(evaluator->CreateJacobian());
  double cost = -1;
  double residuals[11];
  std::vector<double> gradient(reduced_program->NumEffectiveParameters(), 0.0);
  SetSparseMatrixConstant(jacobian.get(), -1);
  ASSERT_TRUE(evaluator->Evaluate(state.data(),
                                  &cost,
                                  residuals,
                                  gradient.data(),
                                  jacobian.get()));

  ProblemCUDA problem_cuda(problem_options);
  CauchyLossCUDA* cauchy_loss_cuda = new CauchyLossCUDA(1.0);
  HuberLossCUDA* huber_loss_cuda = new HuberLossCUDA(1.0);

  problem_cuda.AddResidualBlock<SnavelyReprojectionErrorWithQuaternions, 2, 10, 3>(
      cost_function1,
      cauchy_loss_cuda,
      camera1,
      point1);
  problem_cuda.AddResidualBlock<SnavelyReprojectionErrorWithQuaternions, 2, 10, 3>(
      cost_function2,
      cauchy_loss_cuda,
      camera2,
      point1);
  problem_cuda.AddResidualBlock<SnavelyReprojectionErrorWithQuaternions, 2, 10, 3>(
      cost_function3,
      cauchy_loss_cuda,
      camera1,
      point2);
  problem_cuda.AddResidualBlock<SnavelyReprojectionErrorNoRadialDistortion, 2, 7, 3>(
      cost_function4,
      huber_loss_cuda,
      camera3,
      point1);
  problem_cuda.AddResidualBlock<PointDisplacementError, 3, 3>(
      cost_function5,
      nullptr,
      point1);
  problem_cuda.AddResidualBlock<PointDisplacementError, 3, 3>(
      cost_function6,
      nullptr,
      point2);

  problem_cuda.SetParameterBlockConstant(camera2);
  problem_cuda.SetParameterBlockConstant(point2);
  problem_cuda.SetManifold(camera1, camera_manifold);

  ProblemImpl* problem_impl_cuda = problem_cuda.mutable_problem()->mutable_impl();
  Program* program_cuda = problem_impl_cuda->mutable_program();
  std::unique_ptr<Program> reduced_program_cuda =
      program_cuda->CreateReducedProgram(&removed_parameter_blocks,
                                         &fixed_cost,
                                         &error);

  options.context = problem_cuda.context();
  options.use_cuda = true;
  options.registered_cuda_evaluators = problem_cuda.mutable_registered_cuda_evaluators();
  std::unique_ptr<Evaluator> evaluator_cuda(
      Evaluator::Create(options, reduced_program_cuda.get(), &error));
  std::unique_ptr<SparseMatrix> jacobian_cuda(evaluator_cuda->CreateJacobian());

  double cost_cuda = -1;
  double residuals_cuda[11];
  std::vector<double> gradient_cuda(reduced_program_cuda->NumEffectiveParameters(), 0.0);
  SetSparseMatrixConstant(jacobian_cuda.get(), -1);

  ASSERT_TRUE(evaluator_cuda->Evaluate(state.data(),
                                       &cost_cuda,
                                       residuals_cuda,
                                       gradient_cuda.data(),
                                       jacobian_cuda.get()));

  ConstVectorRef residual_values(residuals, 11);
  ConstVectorRef residual_cuda_values(residuals_cuda, 11);
  ConstVectorRef gradient_values(gradient.data(),
                                 reduced_program->NumEffectiveParameters());
  ConstVectorRef gradient_cuda_values(gradient_cuda.data(),
                                      reduced_program_cuda->NumEffectiveParameters());
  ConstVectorRef jacobian_values(jacobian->values(),
                                 jacobian->num_nonzeros());
  ConstVectorRef jacobian_cuda_values(jacobian_cuda->values(),
                                      jacobian_cuda->num_nonzeros());

  EXPECT_NEAR(cost_cuda, cost, kTolerance);
  EXPECT_TRUE(residual_cuda_values.isApprox(residual_values, kTolerance))
          << "Residuals computed in CUDA:\n"
          << residual_cuda_values
          << "\nResiduals computed on the host:\n"
          << residual_values;
  EXPECT_TRUE(gradient_cuda_values.isApprox(gradient_values, kTolerance))
          << "Gradient computed in CUDA:\n"
          << gradient_cuda_values
          << "\nGradient computed on the host:\n"
          << gradient_values;
  EXPECT_TRUE(jacobian_cuda_values.isApprox(jacobian_values, kTolerance))
          << "Jacobian values computed in CUDA:\n"
          << jacobian_cuda_values
          << "\nJacobian values computed on the host:\n"
          << jacobian_values;

  delete cost_function1;
  delete cost_function2;
  delete cost_function3;
  delete cost_function4;
  delete cost_function5;
  delete cost_function6;
  delete camera_manifold;
}

TEST(EvaluatorCUDA, EvaluateBundleAdjustmentProblemBlockSparseMatrix) {
  // A BlockSparseMatrix will be used to store the Jacobian.
  EvaluateBundleAdjustmentProblem(NO_SPARSE);
}

TEST(EvaluatorCUDA, EvaluateBundleAdjustmentProblemCompressedRowSparseMatrix) {
  // A CompressedRowSparseMatrix will be used to store the Jacobian.
  EvaluateBundleAdjustmentProblem(CUDA_SPARSE);
}

}  // namespace ceres::internal
