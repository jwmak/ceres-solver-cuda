// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2022 Google Inc. All rights reserved.
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
// ProgramEvaluatorCUDA is similar to ProgramEvaluator, except that
// the cost functions are evaluated in parallel on the GPU. The cost
// functions are evaluated using the registered evaluators in the
// RegisteredCUDAEvaluators object.

#ifndef CERES_INTERNAL_PROGRAM_EVALUATOR_CUDA_H_
#define CERES_INTERNAL_PROGRAM_EVALUATOR_CUDA_H_

// This include must come before any #ifndef check on Ceres compile options.
// clang-format off
#include "ceres/internal/config.h"
// clang-format on

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ceres/evaluation_callback.h"
#include "ceres/execution_summary.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/registered_cuda_evaluators.h"
#include "ceres/parameter_block.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"
#include "ceres/small_blas.h"

namespace ceres::internal {

struct NullJacobianFinalizerCUDA {
  void operator()(SparseMatrix* /*jacobian*/, int /*num_parameters*/) {}
};

template <typename JacobianWriter,
          typename JacobianFinalizer = NullJacobianFinalizerCUDA>
class ProgramEvaluatorCUDA final : public Evaluator {
 public:
  ProgramEvaluatorCUDA(const Evaluator::Options& options,
                        Program* program)
      : options_(options),
        program_(program),
        jacobian_writer_(options, program),
        registered_cuda_evaluators_(options.registered_cuda_evaluators),
        num_parameters_(program->NumEffectiveParameters()) {
    // Build data structures that provide the correct locations for the
    // CUDA evaluators to write the output residuals and output jacobians.
    BuildResidualLayout(*program, &residual_layout_);
    jacobian_writer_.CreateJacobianPerResidualLayout(&jacobian_per_residual_layout_,
                                                     &jacobian_per_residual_offsets_,
                                                     &num_jacobian_values_);

    // At this point, the RegisteredCUDAEvaluators object has not been
    // initalialized yet, since the program was not finished re-ordering
    // and removing residual blocks and parameter blocks.
    registered_cuda_evaluators_->Init(program,
                                      residual_layout_,
                                      jacobian_per_residual_layout_,
                                      jacobian_per_residual_offsets_,
                                      num_jacobian_values_);
  }

  // Implementation of Evaluator interface.
  std::unique_ptr<SparseMatrix> CreateJacobian() const final {
    return jacobian_writer_.CreateJacobian();
  }

  bool Evaluate(const Evaluator::EvaluateOptions& evaluate_options,
                const double* state,
                double* cost,
                double* residuals,
                double* gradient,
                SparseMatrix* jacobian) final {
    ScopedExecutionTimer total_timer("Evaluator::Total", &execution_summary_);
    ScopedExecutionTimer call_type_timer(
        gradient == nullptr && jacobian == nullptr ? "Evaluator::Residual"
                                                   : "Evaluator::Jacobian",
                                                    &execution_summary_);

    // The parameters are stateful, so set the state before evaluating.
    if (!program_->StateVectorToParameterBlocks(state)) {
      return false;
    }

    // Notify the user about a new evaluation point if they are interested.
    if (options_.evaluation_callback != nullptr) {
      program_->CopyParameterBlockStateToUserState();
      options_.evaluation_callback->PrepareForEvaluation(
          /*jacobians=*/(gradient != nullptr || jacobian != nullptr),
          evaluate_options.new_evaluation_point);
    }

    double* jacobian_values = jacobian ? jacobian->mutable_values() : nullptr;
    bool abort = registered_cuda_evaluators_->Evaluate(state,
                                                       cost,
                                                       residuals,
                                                       gradient,
                                                       jacobian_values);
    if (!abort)
      return false;

    if (jacobian != nullptr) {
      JacobianFinalizer f;
      f(jacobian, num_parameters_);
    }

    return true;
  }

  bool Plus(const double* state,
            const double* delta,
            double* state_plus_delta) const final {
    return program_->Plus(
        state, delta, state_plus_delta, options_.context, options_.num_threads);
  }

  int NumParameters() const final { return program_->NumParameters(); }
  int NumEffectiveParameters() const final {
    return program_->NumEffectiveParameters();
  }

  int NumResiduals() const final { return program_->NumResiduals(); }

  std::map<std::string, CallStatistics> Statistics() const final {
    return execution_summary_.statistics();
  }

 private:
  static void BuildResidualLayout(const Program& program,
                                  std::vector<int>* residual_layout) {
    const std::vector<ResidualBlock*>& residual_blocks =
        program.residual_blocks();
    residual_layout->resize(program.NumResidualBlocks());
    int residual_pos = 0;
    for (int i = 0; i < residual_blocks.size(); ++i) {
      const int num_residuals = residual_blocks[i]->NumResiduals();
      (*residual_layout)[i] = residual_pos;
      residual_pos += num_residuals;
    }
  }

  Evaluator::Options options_;
  Program* program_;
  JacobianWriter jacobian_writer_;
  std::vector<int> residual_layout_;
  int num_parameters_;
  int num_jacobian_values_;
  ::ceres::internal::ExecutionSummary execution_summary_;

  RegisteredCUDAEvaluators* registered_cuda_evaluators_;
  std::vector<int> jacobian_per_residual_layout_;
  std::vector<int> jacobian_per_residual_offsets_;
};

}  // namespace ceres::internal

#endif  // CERES_INTERNAL_PROGRAM_EVALUATOR_CUDA_H_
