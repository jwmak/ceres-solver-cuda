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
// A container of CUDA evaluators for different types of residual
// blocks. Each type of ResidualBlockCUDA needs a separate templated
// evaluator class to evaluate the residual blocks' cost functions.
// RegisteredCUDAEvaluators registers an evaluator object when a
// residual block that needs that type of evaluator is first added
// to ProblemCUDA. RegisteredCUDAEvaluators stores these evaluators
// in an unordered_map which is indexed using std::type_index.
// This STL class uses type info to create a unique index for
// each type of ResidualBlockCUDA.

#ifndef INCLUDE_CERES_INTERNAL_REGISTERED_CUDA_EVALUATORS_H_
#define INCLUDE_CERES_INTERNAL_REGISTERED_CUDA_EVALUATORS_H_

#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA

#include <memory>
#include <typeindex>
#include <unordered_map>

#include "ceres/internal/cuda_buffer.h"
#include "ceres/internal/export.h"
#include "ceres/internal/parameter_block_cuda.h"

namespace ceres::internal {

class ContextImpl;
class Program;
class ResidualBlock;
class ResidualBlockCUDA;
class ResidualBlockCUDAEvaluator;

class CERES_NO_EXPORT RegisteredCUDAEvaluators {
public:
  // Given a ContextImpl, construct RegisteredCUDAEvaluators.
  //
  // This has the side effect of initializing CUDA in the
  // ContextImpl, if this has not been done previously.
  explicit RegisteredCUDAEvaluators(ContextImpl* context);

  // Iterate through the registered CUDA evaluators
  // for all cost/loss function types and evaluate
  // the set of residual blocks assigned to each one.
  bool Evaluate(const double* state,
                double* cost,
                double* residuals,
                double* gradient,
                double* jacobian_matrix_values);

  // Copy the updated plus jacobians to device memory.
  // Between each evaluation call, the plus jacobians
  // on the host may have changed.
  void UpdatePlusJacobians();

  // Collect all parameter blocks in the program
  // and use them to create a vector of ParameterBlockCUDA
  // which will copied to device memory.
  void SetupParameterBlocks(Program* program);

  // Collect all residual blocks in the program and assign
  // them to the appropriate registered CUDA evaluator.
  void SetupResidualBlocks(Program* program);

  // Set up the parameter blocks and residual blocks in device memory.
  //
  // This method also passes the data structures that contain
  // the correct locations to write jacobians and residuals.
  // Furthermore, this method initializes each registered CUDA evaluator.
  // a process in which each evaluator is assigned its correct
  // residual blocks and each evaluator copies its residual blocks
  // to device memory.
  void Init(Program* program,
            const std::vector<int>& residual_layout,
            const std::vector<int>& jacobian_per_residual_layout,
            const std::vector<int>& jacobian_per_residual_offsets,
            int num_jacobian_values);

  // Return the registered CUDA evaluator for an evaluator type.
  ResidualBlockCUDAEvaluator* GetResidualBlockCUDAEvaluator(
                                     const std::type_index& evaluator_type);


  // Register a ResidualBlockCUDAEvaluator for a residual block type.
  //
  // Given a evaluator type which is determined by cost/loss functors
  // and represented as a hashable std::type_index struct, register
  // a matching ResidualBlockCUDAEvaluator for that type by storing it
  // into an unordered map.
  void RegisterCUDAEvaluator(const std::type_index& evaluator_type,
                             ResidualBlockCUDAEvaluator* residual_block_cuda_evaluator);

private:
  // Adds a residual block to the correct registered cuda evalutor.
  //
  // This is done by getting the correct std::type_index of the derived
  // class by calling std::type_index(typeid(*residual_block_cuda)).
  // Using typeid on the dereferenced pointer will get the type id
  // of the derived class (an AutoDiffResidualBlockCUDAWrapper<...>)
  // instead of the base class of ResidualBlockCUDA.
  void AddResidualBlockCUDA(ResidualBlockCUDA* residual_block_cuda);

  ContextImpl* context_;
  std::unordered_map<std::type_index, std::unique_ptr<ResidualBlockCUDAEvaluator>> evaluators_;
  std::vector<const double*> plus_jacobians_;
  std::vector<int> plus_jacobian_sizes_;
  std::vector<double> h_plus_jacobians_;

  CudaBuffer<ParameterBlockCUDA> d_parameter_blocks_;
  CudaBuffer<ParameterBlockCUDA> d_constant_parameter_blocks_;

  // The states of all active and constant parameter blocks.
  // These are stored contiguously in arrays in device memory,
  // and each ParameterBlockCUDA's state will point to a
  // location in one of these arrays.
  CudaBuffer<double> d_parameters_state_;
  CudaBuffer<double> d_constant_parameters_state_;

  CudaBuffer<double> d_plus_jacobians_;
  CudaBuffer<double> d_residuals_;
  CudaBuffer<int> d_residual_layout_;

  CudaBuffer<double> d_jacobian_values_;
  CudaBuffer<int> d_jacobian_per_residual_layout_;
  CudaBuffer<int> d_jacobian_per_residual_offsets_;
  CudaBuffer<double> d_gradient_;
};

}  // namespace ceres::internal

#endif  // CERES_NO_CUDA

#endif /* INCLUDE_CERES_INTERNAL_REGISTERED_CUDA_EVALUATORS_H_ */
