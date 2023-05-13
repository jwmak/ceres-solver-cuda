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
// The ParameterBlock class implemented in a format compatible with
// the GPU. ParameterBlockCUDA stores information for each parameter
// block and is meant to be accessed primarily in device code. The
// methods in this class are similar to those found in ParameterBlock.

#ifndef INCLUDE_CERES_INTERNAL_PARAMETER_BLOCK_CUDA_H_
#define INCLUDE_CERES_INTERNAL_PARAMETER_BLOCK_CUDA_H_

#include "ceres/internal/cuda_defs.h"
#include "ceres/internal/export.h"

namespace ceres::internal {

class CERES_NO_EXPORT ParameterBlockCUDA {
 public:
  HOST_DEVICE ParameterBlockCUDA() :
   state_(nullptr),
   tangent_size_(0),
   state_offset_(0),
   delta_offset_(0),
   is_constant_(false),
   plus_jacobian_(nullptr) {}

  HOST_DEVICE bool IsConstant() {
    return is_constant_;
  }

  HOST_DEVICE double* GetState() const {
    return state_;
  }

  HOST_DEVICE int GetDeltaOffset() const {
    return delta_offset_;
  }

  HOST_DEVICE int GetStateOffset() const {
    return state_offset_;
  }

  HOST_DEVICE int GetTangentSize() const {
    return tangent_size_;
  }

  HOST_DEVICE double* GetPlusJacobian() const {
    return plus_jacobian_;
  }

  HOST_DEVICE void SetState(double* state) {
    state_ = state;
  }

  HOST_DEVICE void SetDeltaOffset(int delta_offset) {
    delta_offset_ = delta_offset;
  }

  HOST_DEVICE void SetStateOffset(int state_offset) {
    state_offset_ = state_offset;
  }

  HOST_DEVICE void SetTangentSize(int tangent_size) {
    tangent_size_ = tangent_size;
  }

  HOST_DEVICE void SetPlusJacobian(double* plus_jacobian) {
    plus_jacobian_ = plus_jacobian;
  }

  HOST_DEVICE void SetConstant() {
    is_constant_ = true;
  }

  HOST_DEVICE void SetNonConstant() {
    is_constant_ = false;
  }

 private:
  // This should point to a location in device memory.
  double* state_;

  int tangent_size_;
  int state_offset_;
  int delta_offset_;
  bool is_constant_;

  // This should point to a location in device memory.
  double* plus_jacobian_;
};

}  // namespace ceres::internal

#endif /* INCLUDE_CERES_INTERNAL_PARAMETER_BLOCK_CUDA_H_ */
