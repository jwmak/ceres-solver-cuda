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
// Interface to be inherited by the template AutoDiffResidualBlockCUDAEvaluator.
// Inheriting this common interface allows for type erasure while allowing
// different types of residual blocks to have different templated evaluators.

#ifndef INCLUDE_CERES_INTERNAL_RESIDUAL_BLOCK_CUDA_EVALUATOR_H_
#define INCLUDE_CERES_INTERNAL_RESIDUAL_BLOCK_CUDA_EVALUATOR_H_

namespace ceres::internal {

class ResidualBlockCUDA;

class ResidualBlockCUDAEvaluator {
public:
  // Initializes the data structures and variables
  // needed by the evaluator.
  virtual void Init() = 0;

  // Evaluates all the residual blocks assigned to this evaluator
  // in parallel on the GPU.
  //
  // Extra data structures need to be passed to this function, which
  // provide the correct locations to write the output residuals and
  // jacobians.
  virtual bool Evaluate(double* cost,
                        bool output_residuals,
                        double* d_residuals,
                        int* d_residual_layout,
                        double* d_gradient,
                        double* d_jacobian_values,
                        int* d_jacobian_per_residual_layout,
                        int* d_jacobian_per_residual_offsets) = 0;

  // Add a residual block to this evaluator.
  //
  // A dynamic_cast is used to check that the ResidualBlockCUDA is
  // in fact a derived object of the correct type for this evaluator.
  virtual void AddResidualBlock(ResidualBlockCUDA* residual_block_cuda) = 0;

  // Returns the number of residual blocks assigned to this evaluator.
  virtual int NumResidualBlocks() = 0;

  // Resets the evaluator by clearing some buffers.
  //
  // This is only useful if the same ProblemCUDA has finished solving in
  // its current Solver and is passed to a new Solver to continue solving.
  virtual void Reset() = 0;

  virtual ~ResidualBlockCUDAEvaluator() {};
};

}  // namespace ceres::internal

#endif /* INCLUDE_CERES_INTERNAL_RESIDUAL_BLOCK_CUDA_EVALUATOR_H_ */
