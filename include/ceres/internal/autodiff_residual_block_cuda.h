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
// A template, AutoDiffResidualBlockCUDAWrapper<...>, that inherits ResidualBlockCUDA.
// The two most important template parameters, the cost function and the loss
// function, uniquely identify a type for a residual block.
//
// The object being "wrapped" is AutoDiffResidualBlockCUDA, which is also
// created with a template. Since polymorphic objects initialized
// on the host cannot be used on the GPU, AutoDiffResidualBlockCUDA contains
// no virtual functions. The methods in AutoDiffResidualBlockCUDA are also
// compiled for both host and device. Polymorphic behavior is implemented by
// having different instantiations of AutoDiffResidualBlockCUDA share the
// common functions Evaluate() and EvaluateLossFunction().

#ifndef INTERNAL_CERES_AUTODIFF_RESIDUAL_BLOCK_CUDA_H_
#define INTERNAL_CERES_AUTODIFF_RESIDUAL_BLOCK_CUDA_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ceres/autodiff_cost_function_cuda.h"
#include "ceres/cost_function.h"
#include "ceres/internal/cuda_defs.h"
#include "ceres/internal/parameter_block_cuda.h"
#include "ceres/internal/residual_block_cuda.h"
#include "ceres/loss_function_cuda.h"
#include "ceres/types.h"

namespace ceres::internal {

template <typename CostFunctor,
          typename LossFunctionCUDA,
          int kNumResiduals,
          int... Ns>
class AutoDiffResidualBlockCUDA;

template <typename CostFunctor,
          typename LossFunctionCUDA,
          int kNumResiduals,
          int... Ns>
class AutoDiffResidualBlockCUDAWrapper : public ResidualBlockCUDA {
  using AutoDiffResidualBlockCUDA_t =
      AutoDiffResidualBlockCUDA<CostFunctor, LossFunctionCUDA, kNumResiduals, Ns...>;
  using AutoDiffCostFunctionCUDA_t=
      AutoDiffCostFunctionCUDA<CostFunctor, kNumResiduals, Ns...>;

 public:
  AutoDiffResidualBlockCUDAWrapper(const CostFunctor& functor,
                                   LossFunctionCUDA* loss_function)
   : loss_function_{loss_function},
     cost_function_{new AutoDiffCostFunctionCUDA_t(functor)},
     residual_block_{new AutoDiffResidualBlockCUDA_t} {}

  void EvaluateLossFunction(double sq_norm, double out[3]) const {
    loss_function_->Evaluate(sq_norm, out);
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const {
    return cost_function_->Evaluate(parameters, residuals, jacobians);
  }

  // The size of the residual vector returned by this residual function.
  int NumResiduals() const { return kNumResiduals; }

  AutoDiffCostFunctionCUDA_t* GetCostFunction() const {
    return cost_function_.get();
  }

  LossFunctionCUDA* GetLossFunction() const {
    return loss_function_;
  }

  bool HasLossFunction() const {
    return loss_function_ != nullptr;
  }

  int GetGlobalResidualBlockIndex() const override {
    return residual_block_->GetGlobalResidualBlockIndex();
  }

  ParameterBlockCUDA** GetParameterBlocks() const {
    return residual_block_->GetParameterBlocks();
  }

  ParameterBlockCUDA* GetParameterBlock(int index) const {
    return residual_block_->GetParameterBlock(index);
  }

  int GetNumParameterBlocks() const override {
    return sizeof...(Ns);
  }

  void SetGlobalResidualBlockIndex(int index) override {
    residual_block_->SetGlobalResidualBlockIndex(index);
  }

  void SetParameterBlock(int index, ParameterBlockCUDA* parameter_block) {
    residual_block_->SetParameterBlock(index, parameter_block);
  }

  AutoDiffResidualBlockCUDA_t* GetResidualBlock() const {
    return residual_block_.get();
  }

 private:
  std::unique_ptr<AutoDiffResidualBlockCUDA_t> residual_block_;
  std::unique_ptr<AutoDiffCostFunctionCUDA_t> cost_function_;
  LossFunctionCUDA* loss_function_;
};

template <typename CostFunctor,
          typename LossFunctionCUDA,
          int kNumResiduals,
          int... Ns>
class CERES_NO_EXPORT AutoDiffResidualBlockCUDA {
  using AutoDiffCostFunctionCUDA_t =
      AutoDiffCostFunctionCUDA<CostFunctor, kNumResiduals, Ns...>;

 public:
  HOST_DEVICE void EvaluateLossFunction(double sq_norm, double out[3]) const {
    loss_function_->Evaluate(sq_norm, out);
  }

  HOST_DEVICE bool Evaluate(double const* const* parameters,
                            double* residuals,
                            double** jacobians) const {
    return cost_function_->Evaluate(parameters, residuals, jacobians);
  }

  // The size of the residual vector returned by this residual function.
  HOST_DEVICE int NumResiduals() const { return kNumResiduals; }

  HOST_DEVICE int GetGlobalResidualBlockIndex() const {
    return global_residual_block_index_;
  }

  HOST_DEVICE ParameterBlockCUDA** GetParameterBlocks() const {
    return (ParameterBlockCUDA**) parameter_blocks_;
  }

  HOST_DEVICE ParameterBlockCUDA* GetParameterBlock(int index) const {
    return parameter_blocks_[index];
  }

  HOST_DEVICE void SetCostFunction(AutoDiffCostFunctionCUDA_t* cost_function) {
    cost_function_ = cost_function;
  }

  HOST_DEVICE void SetLossFunction(LossFunctionCUDA* loss_function) {
    loss_function_ = loss_function;
  }

  HOST_DEVICE void SetGlobalResidualBlockIndex(int index) {
    global_residual_block_index_ = index;
  }

  HOST_DEVICE void SetParameterBlock(int index, ParameterBlockCUDA* parameter_block) {
    parameter_blocks_[index] = parameter_block;
  }

 private:
  int global_residual_block_index_;
  AutoDiffCostFunctionCUDA_t* cost_function_;
  LossFunctionCUDA* loss_function_;
  ParameterBlockCUDA* parameter_blocks_[sizeof...(Ns)];
};

}  // namespace ceres::internal

//#include "ceres/internal/reenable_warnings.h"


#endif /* INTERNAL_CERES_AUTODIFF_RESIDUAL_BLOCK_CUDA_H_ */
