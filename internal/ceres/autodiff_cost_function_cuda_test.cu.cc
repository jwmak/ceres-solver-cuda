// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2019 Google Inc. All rights reserved.
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

#include "ceres/autodiff_cost_function_cuda.h"

#include <memory>

#include "ceres/array_utils.h"
#include "ceres/internal/cuda_buffer.h"
#include "ceres/internal/cuda_defs.h"
#include "gtest/gtest.h"

namespace ceres::internal {

class BinaryScalarCost {
 public:
  HOST_DEVICE explicit BinaryScalarCost(double a) : a_(a) {}
  template <typename T>
  HOST_DEVICE bool operator()(const T* const x, const T* const y, T* cost) const {
    cost[0] = x[0] * y[0] + x[1] * y[1] - T(a_);
    return true;
  }

 private:
  double a_;
};

__global__ void BilinearDifferentiationTestKernel(double* residuals,
                                                  double** jacobians) {
  BinaryScalarCost* functor = new BinaryScalarCost(1.0);
  AutoDiffCostFunctionCUDA<BinaryScalarCost, 1, 2, 2>* cost_function =
      new AutoDiffCostFunctionCUDA<BinaryScalarCost, 1, 2, 2>(
          *functor);

  auto** parameters = new double*[2];
  parameters[0] = new double[2];
  parameters[1] = new double[2];

  parameters[0][0] = 1;
  parameters[0][1] = 2;

  parameters[1][0] = 3;
  parameters[1][1] = 4;

  cost_function->Evaluate(parameters, residuals, jacobians);

  delete[] parameters[0];
  delete[] parameters[1];
  delete[] parameters;
  delete cost_function;
  delete functor;
}

TEST(AutodiffCostFunctionCUDA, BilinearDifferentiationTest) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  auto** jacobians = new double*[2];
  jacobians[0] = new double[2];
  jacobians[1] = new double[2];

  double residuals = 0.0;

  CudaBuffer<double*> jacobians_device(&context, 2);
  CudaBuffer<double> block_jacobian0_device(&context, 2);
  CudaBuffer<double> block_jacobian1_device(&context, 2);
  CudaBuffer<double> residuals_device(&context, 1);
  double* temp[2] = { block_jacobian0_device.data(), block_jacobian1_device.data() };
  jacobians_device.CopyFromCpu(temp, 2);

  BilinearDifferentiationTestKernel<<<1, 1, 0, context.DefaultStream()>>>(residuals_device.data(),
                                                                          nullptr);
  residuals_device.CopyToCpu(&residuals, 1);
  EXPECT_EQ(10.0, residuals);

  residuals = 0.0;
  residuals_device.Zero();
  BilinearDifferentiationTestKernel<<<1, 1, 0, context.DefaultStream()>>>(residuals_device.data(),
                                                                          jacobians_device.data());
  residuals_device.CopyToCpu(&residuals, 1);
  block_jacobian0_device.CopyToCpu(jacobians[0], 2);
  block_jacobian1_device.CopyToCpu(jacobians[1], 2);

  EXPECT_EQ(10.0, residuals);
  EXPECT_EQ(3, jacobians[0][0]);
  EXPECT_EQ(4, jacobians[0][1]);
  EXPECT_EQ(1, jacobians[1][0]);
  EXPECT_EQ(2, jacobians[1][1]);

  delete[] jacobians[0];
  delete[] jacobians[1];
  delete[] jacobians;
}

struct TenParameterCost {
  template <typename T>
  HOST_DEVICE bool operator()(const T* const x0,
                              const T* const x1,
                              const T* const x2,
                              const T* const x3,
                              const T* const x4,
                              const T* const x5,
                              const T* const x6,
                              const T* const x7,
                              const T* const x8,
                              const T* const x9,
                              T* cost) const {
    cost[0] = *x0 + *x1 + *x2 + *x3 + *x4 + *x5 + *x6 + *x7 + *x8 + *x9;
    return true;
  }
};

__global__ void ManyParameterAutodiffInstantiatesTestKernel(double* residuals,
                                                            double** jacobians) {
  TenParameterCost* functor = new TenParameterCost;
  AutoDiffCostFunctionCUDA<TenParameterCost,
                           1,
                           1,
                           1,
                           1,
                           1,
                           1,
                           1,
                           1,
                           1,
                           1,
                           1>* cost_function =
      new AutoDiffCostFunctionCUDA<TenParameterCost,
                                   1,
                                   1,
                                   1,
                                   1,
                                   1,
                                   1,
                                   1,
                                   1,
                                   1,
                                   1,
                                   1>(*functor);

  auto** parameters = new double*[10];
  for (int i = 0; i < 10; ++i) {
    parameters[i] = new double[1];
    parameters[i][0] = i;
  }

  cost_function->Evaluate(parameters, residuals, jacobians);

  for (int i = 0; i < 10; ++i) {
    delete[] parameters[i];
  }
  delete[] parameters;
  delete cost_function;
  delete functor;
}

TEST(AutodiffCostFunctionCUDA, ManyParameterAutodiffInstantiates) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  double residuals = 0.0;
  CudaBuffer<double> residuals_device(&context, 1);

  double* temp[10];
  CudaBuffer<double> block_jacobians_device(&context, 10);
  for (int i = 0; i < 10; ++i) {
    temp[i] = block_jacobians_device.data() + i;
  }

  CudaBuffer<double*> jacobians_device(&context, 10);
  jacobians_device.CopyFromCpu(temp, 10);

  ManyParameterAutodiffInstantiatesTestKernel<<<1, 1, 0, context.DefaultStream()>>>(residuals_device.data(),
                                                                                    nullptr);
  residuals_device.CopyToCpu(&residuals, 1);
  EXPECT_EQ(45.0, residuals);

  residuals = 0.0;
  residuals_device.Zero();
  ManyParameterAutodiffInstantiatesTestKernel<<<1, 1, 0, context.DefaultStream()>>>(residuals_device.data(),
                                                                                    jacobians_device.data());
  residuals_device.CopyToCpu(&residuals, 1);
  EXPECT_EQ(residuals, 45.0);

  double* block_jacobians = new double[10];
  block_jacobians_device.CopyToCpu(block_jacobians, 10);

  auto** jacobians = new double*[10];
  for (int i = 0; i < 10; ++i) {
    jacobians[i] = &block_jacobians[i];
  }
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(1.0, jacobians[i][0]);
  }

  delete[] block_jacobians;
  delete[] jacobians;
}

struct OnlyFillsOneOutputFunctor {
  template <typename T>
  HOST_DEVICE bool operator()(const T* x, T* output) const {
    output[0] = x[0];
    return true;
  }
};

using booltype = uint8_t;

__device__ void InvalidateArrayDevice(const int64_t size, double* x) {
  if (x != nullptr) {
    for (int64_t i = 0; i < size; ++i) {
      x[i] = kImpossibleValue;
    }
  }
}

__global__ void PartiallyFilledResidualShouldFailEvaluationTestKernel(double* residuals,
                                                                      double** jacobians,
                                                                      booltype* return_val) {
  OnlyFillsOneOutputFunctor* functor = new OnlyFillsOneOutputFunctor;
  AutoDiffCostFunctionCUDA<OnlyFillsOneOutputFunctor, 2, 1>* cost_function =
      new AutoDiffCostFunctionCUDA<OnlyFillsOneOutputFunctor, 2, 1>(*functor);

  double parameter = 1.0;
  double* parameters[] = {&parameter};

  InvalidateArrayDevice(2, jacobians[0]);
  InvalidateArrayDevice(2, residuals);
  *return_val = cost_function->Evaluate(parameters, residuals, jacobians);

  delete cost_function;
  delete functor;
}

TEST(AutoDiffCostFunctionCUDA, PartiallyFilledResidualShouldFailEvaluation) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  double jacobian[2];
  double residuals[2];
  booltype return_val = false;

  CudaBuffer<double> block_jacobian_device(&context, 2);
  CudaBuffer<double> residuals_device(&context, 2);
  CudaBuffer<double*> jacobians_device(&context, 1);
  CudaBuffer<booltype> return_val_device(&context, 1);

  double* jacobians[1];
  jacobians[0] = block_jacobian_device.data();
  jacobians_device.CopyFromCpu(jacobians, 1);

  PartiallyFilledResidualShouldFailEvaluationTestKernel<<<1, 1, 0, context.DefaultStream()>>>(residuals_device.data(),
                                                                                              jacobians_device.data(),
                                                                                              return_val_device.data());
  block_jacobian_device.CopyToCpu(jacobian, 2);
  residuals_device.CopyToCpu(residuals, 2);
  return_val_device.CopyToCpu(&return_val, 1);

  EXPECT_TRUE(return_val);
  EXPECT_FALSE(IsArrayValid(2, jacobian));
  EXPECT_FALSE(IsArrayValid(2, residuals));
}

}  // namespace ceres::internal
