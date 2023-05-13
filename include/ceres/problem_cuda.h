// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2021 Google Inc. All rights reserved.
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
// A class used to represent non-linear least squares problems and use
// CUDA to evaluate and auto-differentiate cost functions in parallel
// on the GPU. The class is also a wrapper around ceres::Problem to
// take advantage of the existing framework used to represent a problem.

#ifndef CERES_PUBLIC_PROBLEM_CUDA_H_
#define CERES_PUBLIC_PROBLEM_CUDA_H_

#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "ceres/internal/config.h"
#include "ceres/autodiff_cost_function.h"
#include "ceres/autodiff_cost_function_cuda.h"
#include "ceres/context.h"
#include "ceres/internal/autodiff_residual_block_cuda.h"
#include "ceres/internal/autodiff_residual_block_cuda_evaluator.h"
#include "ceres/internal/disable_warnings.h"
#include "ceres/internal/port.h"
#include "ceres/internal/registered_cuda_evaluators.h"
#include "ceres/loss_function_cuda.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "ceres/types.h"
#include "glog/logging.h"

namespace ceres {

class CostFunction;
class EvaluationCallback;
class LossFunction;
class Manifold;
class Solver;
struct CRSMatrix;

namespace internal {
class Preprocessor;
class ProblemImpl;
class ParameterBlock;
class ResidualBlock;
}  // namespace internal

// A ResidualBlockId is an opaque handle clients can use to remove residual
// blocks from a Problem after adding them.
using ResidualBlockId = internal::ResidualBlock*;

class ProblemCUDA {
 public:

  // The default constructor is equivalent to the invocation
  // ProblemCUDA(Problem::Options()).
  ProblemCUDA()
    : problem_(new Problem),
      registered_cuda_evaluators_(new internal::RegisteredCUDAEvaluators(problem_->context())) {}
  explicit ProblemCUDA(const Problem::Options& options)
    : problem_(new Problem(options)),
      registered_cuda_evaluators_(new internal::RegisteredCUDAEvaluators(problem_->context())) {}
  ProblemCUDA(ProblemCUDA&&) = default;
  ProblemCUDA& operator=(ProblemCUDA&&) = default;
  ~ProblemCUDA() = default;

  ProblemCUDA(const ProblemCUDA&) = delete;
  ProblemCUDA& operator=(const ProblemCUDA&) = delete;

  // Adds a residual block to ProblemCUDA.
  //
  // Unlike Problem::AddResidualBlock(), this method requires the user
  // to pass the cost functor, loss functor, number of residuals,
  // and parameter block sizes as template parameters. These will
  // be used to register the appropriate CUDA evaluators for all
  // given types of residual blocks.
  template <typename CostFunctor,
            int kNumResiduals, // Number of residuals
            int... Ns,         // Number of parameters in each parameter block.
            typename... Ts,
            typename LossFunctionCUDA>
  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                   LossFunctionCUDA* loss_function,
                                   double* x0,
                                   Ts*... xs) {
    // If the user passes a nullptr for loss_function,
    // then the function should be called again with a nullptr type
    // regardless of the type of LossFunctionCUDA*. For example,
    // if the user passes HuberLossCUDA* loss_function = nullptr
    // as the loss function, AddResidualBlock should be called again
    // with a plain nullptr.
    if (!loss_function) {
      return AddResidualBlock<CostFunctor,
                              kNumResiduals,
                              Ns...>(cost_function,
                                     nullptr,
                                     x0,
                                     xs...);
    } else {
      return InternalAddResidualBlock<CostFunctor,
                                      kNumResiduals,
                                      Ns...>(cost_function,
                                             loss_function,
                                             x0,
                                             xs...);
    }
  }

  // Add a residual block with a nullptr passed as the loss function.
  //
  // If a nullptr is passed as as the loss function, then TrivialLossCUDA
  // is used as the template argument for LossFunctionCUDA.
  template <typename CostFunctor,
            int kNumResiduals,
            int... Ns,
            typename... Ts>
  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                   std::nullptr_t,
                                   double* x0,
                                   Ts*... xs) {
    return InternalAddResidualBlock<CostFunctor,
                                    kNumResiduals,
                                    Ns...>(cost_function,
                                           (TrivialLossCUDA*) nullptr,
                                           x0,
                                           xs...);
  }

  internal::ContextImpl* context() {
    return problem_->context();
  }

  // Add a parameter block with appropriate size to the problem. Repeated calls
  // with the same arguments are ignored. Repeated calls with the same double
  // pointer but a different size will result in a crash.
  void AddParameterBlock(double* values, int size) {
    problem_->AddParameterBlock(values, size);
  }

  // Add a parameter block with appropriate size and Manifold to the
  // problem. It is okay for manifold to be nullptr.
  //
  // Repeated calls with the same arguments are ignored. Repeated calls
  // with the same double pointer but a different size results in a crash
  // (unless Solver::Options::disable_all_safety_checks is set to true).
  //
  // Repeated calls with the same double pointer and size but different Manifold
  // is equivalent to calling SetManifold(manifold), i.e., any previously
  // associated Manifold object will be replaced with the manifold.
  void AddParameterBlock(double* values, int size, Manifold* manifold) {
    problem_->AddParameterBlock(values, size, manifold);
  }

  // Remove a parameter block from the problem. The Manifold of the parameter
  // block, if it exists, will persist until the deletion of the problem
  // (similar to cost/loss functions in residual block removal). Any residual
  // blocks that depend on the parameter are also removed, as described above
  // in RemoveResidualBlock().
  //
  // If Problem::Options::enable_fast_removal is true, then the removal is fast
  // (almost constant time). Otherwise, removing a parameter block will incur a
  // scan of the entire Problem object.
  //
  // WARNING: Removing a residual or parameter block will destroy the implicit
  // ordering, rendering the jacobian or residuals returned from the solver
  // uninterpretable. If you depend on the evaluated jacobian, do not use
  // remove! This may change in a future release.
  void RemoveParameterBlock(const double* values) {
    problem_->RemoveParameterBlock(values);
  }

  // Remove a residual block from the problem. Any parameters that the residual
  // block depends on are not removed. The cost and loss functions for the
  // residual block will not get deleted immediately; won't happen until the
  // problem itself is deleted.
  //
  // WARNING: Removing a residual or parameter block will destroy the implicit
  // ordering, rendering the jacobian or residuals returned from the solver
  // uninterpretable. If you depend on the evaluated jacobian, do not use
  // remove! This may change in a future release.
  void RemoveResidualBlock(ResidualBlockId residual_block) {
    problem_->RemoveResidualBlock(residual_block);
  }

  // Hold the indicated parameter block constant during optimization.
  void SetParameterBlockConstant(const double* values) {
    problem_->SetParameterBlockConstant(values);
  }

  // Allow the indicated parameter block to vary during optimization.
  void SetParameterBlockVariable(double* values) {
    problem_->SetParameterBlockVariable(values);
  }

  // Returns true if a parameter block is set constant, and false otherwise. A
  // parameter block may be set constant in two ways: either by calling
  // SetParameterBlockConstant or by associating a Manifold with a zero
  // dimensional tangent space with it.
  bool IsParameterBlockConstant(const double* values) const {
    return problem_->IsParameterBlockConstant(values);
  }

  // Set the Manifold for the parameter block. Calling SetManifold with nullptr
  // will clear any previously set Manifold for the parameter block.
  //
  // Repeated calls will result in any previously associated Manifold object to
  // be replaced with the manifold.
  //
  // The manifold is owned by the Problem by default (See Problem::Options to
  // override this behaviour).
  //
  // It is acceptable to set the same Manifold for multiple parameter blocks.
  void SetManifold(double* values, Manifold* manifold) {
    problem_->SetManifold(values, manifold);
  }
  // Get the Manifold object associated with this parameter block.
  //
  // If there is no Manifold object associated then nullptr is returned.
  const Manifold* GetManifold(const double* values) const {
    return problem_->GetManifold(values);
  }

  // Returns true if a Manifold is associated with this parameter block, false
  // otherwise.
  bool HasManifold(const double* values) const {
    return problem_->HasManifold(values);
  }
  // Set the lower/upper bound for the parameter at position "index".
  void SetParameterLowerBound(double* values,
                                       int index,
                                       double lower_bound) {
    problem_->SetParameterLowerBound(values, index, lower_bound);
  }

  void SetParameterUpperBound(double* values,
                                       int index,
                                       double upper_bound) {
    problem_->SetParameterUpperBound(values, index, upper_bound);
  }

  // Get the lower/upper bound for the parameter at position "index". If the
  // parameter is not bounded by the user, then its lower bound is
  // -std::numeric_limits<double>::max() and upper bound is
  // std::numeric_limits<double>::max().
  double GetParameterUpperBound(const double* values, int index) const {
    return problem_->GetParameterUpperBound(values, index);
  }

  double GetParameterLowerBound(const double* values, int index) const {
    return problem_->GetParameterLowerBound(values, index);
  }
  // Number of parameter blocks in the problem. Always equals
  // parameter_blocks().size() and parameter_block_sizes().size().
  int NumParameterBlocks() const { return problem_->NumParameterBlocks(); }

  // The size of the parameter vector obtained by summing over the sizes of all
  // the parameter blocks.
  int NumParameters() const { return problem_->NumParameters(); }

  // Number of residual blocks in the problem. Always equals
  // residual_blocks().size().
  int NumResidualBlocks() const { return problem_->NumResidualBlocks(); }

  // The size of the residual vector obtained by summing over the sizes of all
  // of the residual blocks.
  int NumResiduals() const { return problem_->NumResiduals(); }

  // The size of the parameter block.
  int ParameterBlockSize(const double* values) const {
    return problem_->ParameterBlockSize(values);
  }

  // The dimension of the tangent space of the Manifold for the parameter block.
  // If there is no Manifold associated with this parameter block, then
  // ParameterBlockTangentSize = ParameterBlockSize.
  int ParameterBlockTangentSize(const double* values) const {
    return problem_->ParameterBlockTangentSize(values);
  }

  // Is the given parameter block present in this problem or not?
  bool HasParameterBlock(const double* values) const {
    return problem_->HasParameterBlock(values);
  }

  // Fills the passed parameter_blocks vector with pointers to the parameter
  // blocks currently in the problem. After this call, parameter_block.size() ==
  // NumParameterBlocks.
  void GetParameterBlocks(std::vector<double*>* parameter_blocks) const {
    problem_->GetParameterBlocks(parameter_blocks);
  }

  // Fills the passed residual_blocks vector with pointers to the residual
  // blocks currently in the problem. After this call, residual_blocks.size() ==
  // NumResidualBlocks.
  void GetResidualBlocks(
      std::vector<ResidualBlockId>* residual_blocks) const {
    problem_->GetResidualBlocks(residual_blocks);
  }

  // Get all the parameter blocks that depend on the given residual block.
  void GetParameterBlocksForResidualBlock(
      const ResidualBlockId residual_block,
      std::vector<double*>* parameter_blocks) const {
    problem_->GetParameterBlocksForResidualBlock(residual_block, parameter_blocks);
  }

  // Get the CostFunction for the given residual block.
  const CostFunction* GetCostFunctionForResidualBlock(
      const ResidualBlockId residual_block) const {
    return problem_->GetCostFunctionForResidualBlock(residual_block);
  }

  // Get the LossFunction for the given residual block. Returns nullptr
  // if no loss function is associated with this residual block.
  const LossFunction* GetLossFunctionForResidualBlock(
      const ResidualBlockId residual_block) const {
    return problem_->GetLossFunctionForResidualBlock(residual_block);
  }

  // Get all the residual blocks that depend on the given parameter block.
  //
  // If Problem::Options::enable_fast_removal is true, then getting the residual
  // blocks is fast and depends only on the number of residual
  // blocks. Otherwise, getting the residual blocks for a parameter block will
  // incur a scan of the entire Problem object.
  void GetResidualBlocksForParameterBlock(
      const double* values, std::vector<ResidualBlockId>* residual_blocks) const {
    problem_->GetResidualBlocksForParameterBlock(values, residual_blocks);
  }

  bool Evaluate(const Problem::EvaluateOptions& evaluate_options,
                double* cost,
                std::vector<double>* residuals,
                std::vector<double>* gradient,
                CRSMatrix* jacobian) {
    return problem_->Evaluate(evaluate_options, cost, residuals, gradient, jacobian);
  }

  bool EvaluateResidualBlock(ResidualBlockId residual_block_id,
                             bool apply_loss_function,
                             double* cost,
                             double* residuals,
                             double** jacobians) const {
    return problem_->EvaluateResidualBlock(residual_block_id,
                                           apply_loss_function,
                                           cost,
                                           residuals,
                                           jacobians);
  }

  bool EvaluateResidualBlockAssumingParametersUnchanged(
      ResidualBlockId residual_block_id,
      bool apply_loss_function,
      double* cost,
      double* residuals,
      double** jacobians) const {
    return problem_->EvaluateResidualBlock(residual_block_id,
                                           apply_loss_function,
                                           cost,
                                           residuals,
                                           jacobians);
  }

  // Returns reference to the options with which the Problem was constructed.
  const Problem::Options& options() const { return problem_->options(); }

  // Returns pointer to Problem.
  Problem* mutable_problem() { return problem_.get(); };

  // Returns pointer to RegisteredCUDAEvaluators.
  internal::RegisteredCUDAEvaluators* mutable_registered_cuda_evaluators() {
    return registered_cuda_evaluators_.get();
  }

 private:
  // Once the correct cost and loss functors have been deduced,
  // add the residual block to the problem.
  //
  // For each cost function, we create a new AutoDiffCostFunctionCUDAWrapper<...>
  // object that will later be copied to device memory. A dynamic_cast is first used
  // to check that the passed cost function is indeed an AutoDiffCostFunction, since
  // these are the only cost function types that are currently supported. We create
  // a new ResidualBlockCUDA, and use std::type_index(typeid(*residual_block_cuda))
  // to generate a hashable type struct based on the cost/loss functors, the number
  // of residuals, and the parameter block sizes. If the type of the residual block
  // is seen for the first time, a new CUDA evaluator (AutoDiffResidualBlockCUDAEvaluator)
  // for that type is registered. At the end, Problem::AddResidualBlockCUDA() is
  // called to associate the newly created ResidualBlockCUDA with a
  // ceres::internal::ResidualBlock that will be created in Problem.
  template <typename CostFunctor,
            int kNumResiduals,
            int... Ns,
            typename... Ts,
            typename LossFunctionCUDA>
  ResidualBlockId InternalAddResidualBlock(CostFunction* cost_function,
                                           LossFunctionCUDA* loss_function,
                                           double* x0,
                                           Ts*... xs) {
    static_assert(kNumResiduals != DYNAMIC,
                  "Can't use the CUDA evaluator if the number of "
                  "residuals is set to ceres::DYNAMIC.");

    using ResidualBlockCUDAType =
        internal::AutoDiffResidualBlockCUDAWrapper<CostFunctor, LossFunctionCUDA, kNumResiduals, Ns...>;

    const std::array<double*, sizeof...(Ts) + 1> parameter_blocks{{x0, xs...}};
    AutoDiffCostFunction<CostFunctor, kNumResiduals, Ns...>* autodiff_cost_function
     = nullptr;

    try {
      autodiff_cost_function =
        dynamic_cast<AutoDiffCostFunction<CostFunctor, kNumResiduals, Ns...>*>(cost_function);
    } catch (const std::exception& e){
      LOG(ERROR) << "Unable to dynamic_cast cost function to 'AutoDiffCostFunction'."
          "The evaluation of cost functions using CUDA is only supported for"
          "cost functions of type 'AutoDiffCostFunction'\n";
    }

    ResidualBlockCUDAType* residual_block_cuda
     = new ResidualBlockCUDAType(autodiff_cost_function->functor(), loss_function);

    if (loss_function && problem_->options().loss_function_ownership == TAKE_OWNERSHIP) {
      if (loss_function_ptrs_.find(loss_function) == loss_function_ptrs_.end()) {
        loss_function_ptrs_[loss_function] =
            std::unique_ptr<LossFunctionCUDABase>(loss_function);
      }
    }

    std::type_index evaluator_type
     = std::type_index(typeid(*residual_block_cuda));

    if (!registered_cuda_evaluators_->GetResidualBlockCUDAEvaluator(evaluator_type)) {
      registered_cuda_evaluators_->RegisterCUDAEvaluator(evaluator_type,
          new internal::AutoDiffResidualBlockCUDAEvaluator<CostFunctor, LossFunctionCUDA, kNumResiduals, Ns...>(context()));
    }

    return problem_->AddResidualBlockCUDA(residual_block_cuda,
                                          cost_function,
                                          parameter_blocks.data(),
                                          static_cast<int>(parameter_blocks.size()));
  }

  // The wrapped Problem.
  std::unique_ptr<Problem> problem_;

  std::unique_ptr<internal::RegisteredCUDAEvaluators> registered_cuda_evaluators_;

  // If ProblemCUDA takes ownership of loss functions, the pointers
  // are stored here and automatically deleted upon the destruction
  // of ProblemCUDA. An unordered_map is used since the user is allowed
  // to assign the same loss function to multiple residual blocks.
  std::unordered_map<LossFunctionCUDABase*, std::unique_ptr<LossFunctionCUDABase>> loss_function_ptrs_;
};


// Helper function which avoids going through the interface.
CERES_EXPORT static void Solve(const Solver::Options& options,
                               ProblemCUDA* problem_cuda,
                               Solver::Summary* summary) {
  Solver solver;
  Solver::Options options_with_cuda = options;
  options_with_cuda.use_cuda_for_evaluator = true;

  // The pointer to RegisteredCUDAEvaluators is passed
  // in the solver options.
  options_with_cuda.registered_cuda_evaluators =
      problem_cuda->mutable_registered_cuda_evaluators();
  solver.Solve(options_with_cuda, problem_cuda->mutable_problem(), summary);
}

}  // namespace ceres

#include "ceres/internal/reenable_warnings.h"

#endif  // CERES_NO_CUDA

#endif  // CERES_PUBLIC_PROBLEM_CUDA_H_
