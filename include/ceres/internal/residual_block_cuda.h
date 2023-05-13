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
// Interface for residual blocks that will be evaluated in device code.
// This interface is inherited by the template AutoDiffResidualBlockCUDAWrapper.
// Inheriting this common base class allows for type erasure while allowing
// different residual blocks to have different cost/loss functions as
// template parameters.


#ifndef INCLUDE_CERES_INTERNAL_RESIDUAL_BLOCK_CUDA_H_
#define INCLUDE_CERES_INTERNAL_RESIDUAL_BLOCK_CUDA_H_

#include "ceres/internal/export.h"

namespace ceres::internal {

class ParameterBlockCUDA;

class CERES_NO_EXPORT ResidualBlockCUDA {
public:
  // Evaluate the residual block's loss function.
  virtual void EvaluateLossFunction(double sq_norm, double out[3]) const = 0;

  // Returns the residual block's position in the array storing all
  // of the program's residual blocks.
  virtual int GetGlobalResidualBlockIndex() const = 0;

  // Returns true if there is a loss function assigned to this residual block.
  virtual bool HasLossFunction() const = 0;

  // Returns a pointer to a buffer containing one or more ParameterBlock*,
  // which are the parameter blocks for this residual block.
  virtual ParameterBlockCUDA** GetParameterBlocks() const = 0;

  // Returns the nth parameter block where n is given by an index.
  virtual ParameterBlockCUDA* GetParameterBlock(int index) const = 0;

  // Returns the number of parameter blocks.
  virtual int GetNumParameterBlocks() const = 0;

  // Sets the index variable that stores this residual block's
  // position in the array storing all of the program's residual
  // blocks.
  virtual void SetGlobalResidualBlockIndex(int index) = 0;

  // Sets the nth parameter block pointer to point to a ParameterBlockCUDA,
  // where n is given by an index. The ParameterBlockCUDA should be
  // residing in device memory.
  virtual void SetParameterBlock(int index,
                                 ParameterBlockCUDA* parameter_block) = 0;

  virtual ~ResidualBlockCUDA() {}
};

}  // namespace ceres::internal

#endif /* INCLUDE_CERES_INTERNAL_RESIDUAL_BLOCK_CUDA_H_ */
