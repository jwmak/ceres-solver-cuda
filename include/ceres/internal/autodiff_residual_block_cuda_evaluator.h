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
//
// Class containing the primary data and functions for evaluating
// cost functions in parallel on the GPU. The class sets up data,
// including input and output buffers, and calls the CUDA kernels
// that will evaluate the residuals, jacobians, and gradient values.

#ifndef INTERNAL_CERES_PROGRAM_EVALUATOR_HOST_H_
#define INTERNAL_CERES_PROGRAM_EVALUATOR_HOST_H_

#include <algorithm>
#include <unordered_map>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <typeindex>
#include <vector>

#include "ceres/autodiff_cost_function_cuda.h"
#include "ceres/internal/autodiff_residual_block_cuda.h"
#include "ceres/internal/cuda_buffer.h"
#include "ceres/internal/cuda_evaluator_kernel.h"
#include "ceres/internal/residual_block_cuda_evaluator.h"
#include "ceres/loss_function_cuda.h"

namespace ceres::internal {

// Number of threads per block for the CUDA kernel.
constexpr int kCudaBlockSize = 256;

template <typename CostFunctor,
          typename LossFunctionCUDA,
          int kNumResiduals,  // Number of residuals
          int... Ns>          // Number of parameters in each parameter block.
class AutoDiffResidualBlockCUDAEvaluator : public ResidualBlockCUDAEvaluator {
  using AutoDiffCostFunctionCUDA_t=
      AutoDiffCostFunctionCUDA<CostFunctor, kNumResiduals, Ns...>;
  using ResidualBlockCUDA_t =
      AutoDiffResidualBlockCUDAWrapper<CostFunctor, LossFunctionCUDA, kNumResiduals, Ns...>;
  using AutoDiffResidualBlockCUDA_t =
      AutoDiffResidualBlockCUDA<CostFunctor, LossFunctionCUDA, kNumResiduals, Ns...>;

public:
  explicit AutoDiffResidualBlockCUDAEvaluator(ContextImpl* context) :
     context_(context),
     apply_loss_function_{false},
     is_device_storage_initialized_ {false},
     d_evaluate_cost_functions_result_{context},
     d_costs_{context},
     d_local_jacobian_values_{context},
     d_cost_functions_{context},
     d_loss_functions_{context},
     d_residual_blocks_{context} {
    parameter_block_sizes_ = {Ns...};
    // Sets apply_loss_function_ to true for any loss function
    // type other than TrivialLossCUDA, false otherwise.
    SetApplyLossFunctionFlag((LossFunctionCUDA*) nullptr);
  }

  // Iterates through all the residual blocks assigned to
  // this evaluator and packs them into an array to be
  // copied to device memory. The same is done to the
  // cost functions and loss functions, which are stored
  // in arrays separate from the residual blocks. Each
  // residual block is set to point appropiately to its
  // cost function and loss function in device memory,
  // after the cost and loss functions have been assigned
  // space in device memory.
  void SetupResidualBlocksOnDevice() {
    int num_residual_blocks = residual_blocks_.size();

    d_cost_functions_.Reserve(num_residual_blocks);
    if (apply_loss_function_)
      d_loss_functions_.Reserve(num_residual_blocks);

    std::vector<AutoDiffResidualBlockCUDA_t> h_residual_blocks;
    std::vector<AutoDiffCostFunctionCUDA_t> h_cost_functions;
    std::vector<LossFunctionCUDA> h_loss_functions;

    // Using reserve() is faster than using resize()
    // and does not require the objects to have default ctors.
    h_residual_blocks.reserve(num_residual_blocks);
    h_cost_functions.reserve(num_residual_blocks);
    if (apply_loss_function_)
      h_loss_functions.reserve(num_residual_blocks);

    for (int i = 0; i < num_residual_blocks; ++i) {
      ResidualBlockCUDA_t* residual_block_wrapper = residual_blocks_[i];
      AutoDiffResidualBlockCUDA_t* residual_block =
          residual_block_wrapper->GetResidualBlock();

      residual_block->SetCostFunction(d_cost_functions_.data() + i);
      h_cost_functions.push_back(std::move(*(residual_block_wrapper->GetCostFunction())));
      if (apply_loss_function_) {
        residual_block->SetLossFunction(d_loss_functions_.data() + i);
        h_loss_functions.push_back(std::move(*(residual_block_wrapper->GetLossFunction())));
      }

      h_residual_blocks.push_back(std::move(*(residual_block)));
    }
    d_residual_blocks_.CopyFromCpu(h_residual_blocks.data(), num_residual_blocks);
    d_cost_functions_.CopyFromCpu(h_cost_functions.data(), num_residual_blocks);

    if (apply_loss_function_)
      d_loss_functions_.CopyFromCpu(h_loss_functions.data(), num_residual_blocks);
  }

  // Uses template deduction to determine if the
  // loss function type is anything other than
  // TrivialLossCUDA. If this is the case, the
  // flag that determines whether or not to apply
  // the loss function after evaluation is set to true.
  template<typename LossFunction>
  void SetApplyLossFunctionFlag(LossFunction*) {
    apply_loss_function_ = true;
  }

  // Sets the flag to false if the loss function type
  // is TrivialLossCUDA.
  void SetApplyLossFunctionFlag(TrivialLossCUDA*) {
    apply_loss_function_ = false;
  }

  // Reserve space for the CUDA buffers and sets up the residual blocks
  // in device memory.
  void Init() {
    int num_residual_blocks = residual_blocks_.size();
    CHECK(num_residual_blocks != 0) << "Initializing AutoDiffResidualBlockCUDAEvaluator"
        << " containing no residual blocks";

    int num_parameters_per_residual_block = std::accumulate(parameter_block_sizes_.begin(),
                                                            parameter_block_sizes_.end(),
                                                            0);
    int total_jacobian_values = num_residual_blocks *
                                kNumResiduals *
                                num_parameters_per_residual_block;

    // Hold the return value across all cost functions.
    // If all cost functions return true, this value is true, otherwise false.
    d_evaluate_cost_functions_result_.Reserve(1);

    // Reserve space for costs
    d_costs_.Reserve(num_residual_blocks);

    // Reserve space for the jacobian values.
    // Use 2x as much needed space so that the second
    // half can be used for scratch space.
    d_local_jacobian_values_.Reserve(total_jacobian_values * 2);

    SetupResidualBlocksOnDevice();

    is_device_storage_initialized_ = true;
  }

  // Evaluate the residuals blocks in parallel using CUDA.
  bool Evaluate(double* cost,
                bool output_residuals,
                double* d_residuals,
                int* d_residual_layout,
                double* d_gradient,
                double* d_jacobian_values,
                int* d_jacobian_per_residual_layout,
                int* d_jacobian_per_residual_offsets) override {

    CHECK(is_device_storage_initialized_) <<
        "Calling AutoDiffResidualBlockCUDAEvaluator::Evaluate() "
          << "before initializing storage on the device.";

    int num_residual_blocks = residual_blocks_.size();
    int num_parameters_per_residual = std::accumulate(parameter_block_sizes_.begin(),
                                                      parameter_block_sizes_.end(),
                                                      0);
    int total_jacobian_values = num_residual_blocks *
                                kNumResiduals *
                                num_parameters_per_residual;

    // Set up the kernel arguments.
    //
    // For a description of each field, see the documentation
    // for the EvaluatorCUDAKernelArguments struct.
    EvaluatorCUDAKernelArguments<CostFunctor, LossFunctionCUDA, kNumResiduals, Ns...> kernel_args;
    kernel_args.num_residual_blocks = num_residual_blocks;
    kernel_args.num_parameters_per_residual = num_parameters_per_residual;

    kernel_args.residual_blocks = d_residual_blocks_.data();
    kernel_args.evaluate_cost_functions_result = d_evaluate_cost_functions_result_.data();

    kernel_args.costs = d_costs_.data();
    kernel_args.residuals = d_residuals;
    kernel_args.residual_layout = d_residual_layout;

    // Use half of a CUDA Buffer to store jacobian scratch space
    // before multiplying by plus jacobians and use the other half
    // to store jacobian values after multiplying by the plus jacobians.
    kernel_args.local_jacobian_values = d_local_jacobian_values_.data();
    kernel_args.jacobian_scratch = d_local_jacobian_values_.data() + total_jacobian_values;

    kernel_args.global_jacobian_values = d_jacobian_values;
    kernel_args.jacobian_per_residual_layout = d_jacobian_per_residual_layout;
    kernel_args.jacobian_per_residual_offsets = d_jacobian_per_residual_offsets;
    kernel_args.gradient = d_gradient;

    kernel_args.apply_loss_function = apply_loss_function_;

    // A separate flag is needed to indicate whether the residuals
    // were requested after evaluation. This is due to the fact
    // that a buffer for residuals is *always* passed because the
    // evaluator needs space for the output residuals of the cost functions.
    kernel_args.output_residuals = output_residuals;
    kernel_args.output_jacobians = d_jacobian_values != nullptr;
    kernel_args.output_gradient = d_gradient != nullptr;

    bool h_evaluate_cost_functions_result = true;
    d_evaluate_cost_functions_result_.CopyFromCpu(&h_evaluate_cost_functions_result, 1);

    // Calculate the number of thread blocks based on the number of
    // residual blocks and the number of threads per block.
    const int num_blocks = (num_residual_blocks + (kCudaBlockSize - 1))
                            / kCudaBlockSize;

    EvaluateKernel<CostFunctor, LossFunctionCUDA, kNumResiduals, Ns...>
      <<<num_blocks, kCudaBlockSize, 0, context_->DefaultStream()>>>
          (kernel_args);

    d_evaluate_cost_functions_result_.CopyToCpu(&h_evaluate_cost_functions_result,
                                                1);
    // During evaluation, if a single cost function returned false,
    // this function should return false, indicating that evaluation
    // has been aborted.
    if (!h_evaluate_cost_functions_result)
      return false;

    // Sum up the costs from all residual blocks in parallel on the GPU.
    double local_cost = thrust::reduce(thrust::device.on(context_->DefaultStream()),
                                       d_costs_.data(),
                                       d_costs_.data() + num_residual_blocks,
                                       0.0,
                                       thrust::plus<double>());

    // Add the cost from this evaluator to the total cost.
    *cost += local_cost;

    return true;
  }

  // Resets the evaluator by clearing a buffer.
  //
  // This is only useful if the same ProblemCUDA has finished solving in
  // its current Solver and is passed to a new Solver to continue solving.
  void Reset() override {
    is_device_storage_initialized_ = false;
    residual_blocks_.clear();
  }

  int NumResidualBlocks() override { return residual_blocks_.size(); }

  // Add a residual block to this evaluator.
  void AddResidualBlock(ResidualBlockCUDA* residual_block_cuda) override {
    ResidualBlockCUDA_t* residual_block_cuda_type = nullptr;

    try {
      residual_block_cuda_type = dynamic_cast<ResidualBlockCUDA_t*>(residual_block_cuda);
    } catch (const std::exception& e){
      std::type_index residual_type = std::type_index(typeid(*residual_block_cuda));
      std::type_index evaluator_type = std::type_index(typeid(*this));

      LOG(ERROR) << "Unable to dynamic_cast ResidualBlockCUDA of type " << residual_type.name()
                 << " to AutoDiffResidualBlockCUDAEvaluator of type " << evaluator_type.name();
    }

    residual_blocks_.push_back(residual_block_cuda_type);
  }

private:
  ContextImpl* context_;
  std::vector<int> parameter_block_sizes_;

  // The residual blocks assigned to this evaluator.
  std::vector<ResidualBlockCUDA_t*> residual_blocks_;

  // Flag to determine whether or not to apply a loss function.
  bool apply_loss_function_;

  // Flag that tracks whether Init() has been called.
  bool is_device_storage_initialized_;

  // A single bool in device memory that is set to false
  // if any cost functions return false during evaluation.
  CudaBuffer<bool> d_evaluate_cost_functions_result_;

  // The resulting costs of evaluating all residual blocks
  // stored in device memory.
  CudaBuffer<double> d_costs_;

  // Stores the intermediate jacobian values prior to writing
  // them to the final jacobian matrix.
  CudaBuffer<double> d_local_jacobian_values_;

  // Cost functions in device memory.
  CudaBuffer<AutoDiffCostFunctionCUDA_t> d_cost_functions_;

  // Loss functions in device memory.
  CudaBuffer<LossFunctionCUDA> d_loss_functions_;

  // Residual blocks in device memory.
  CudaBuffer<AutoDiffResidualBlockCUDA_t> d_residual_blocks_;
};

}  // namespace ceres::internal

#endif /* INTERNAL_CERES_PROGRAM_EVALUATOR_HOST_H_ */
