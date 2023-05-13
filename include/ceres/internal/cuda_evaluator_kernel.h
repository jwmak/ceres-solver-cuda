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
// The CUDA kernel that evaluates residual blocks in parallel, along with
// its supporting device functions.

#ifndef INTERNAL_CERES_PROGRAM_EVALUATOR_HOST_KERNELS_H_
#define INTERNAL_CERES_PROGRAM_EVALUATOR_HOST_KERNELS_H_

#include "ceres/autodiff_cost_function_cuda.h"
#include "ceres/internal/corrector.h"
#include "ceres/internal/eigen.h"

#include "cuda_runtime.h"

namespace ceres::internal {

// Struct to hold the kernel arguments to avoid
// having to pass too many arguments to the kernel
template<typename CostFunctor, typename LossFunctionCUDA, int kNumResiduals, int... Ns>
struct EvaluatorCUDAKernelArguments {
  // The total number of residual blocks to be evaluated by the kernel.
  int num_residual_blocks;

  // The total number of parameters per residual block.
  int num_parameters_per_residual;

  // The array of residual blocks.
  AutoDiffResidualBlockCUDA<CostFunctor, LossFunctionCUDA, kNumResiduals, Ns...>* residual_blocks;

  // The return value of the cost functions.
  // If all cost functions evaluate to true, then this
  // value is set to true. If a single cost function
  // evaluation fails, this value is set to false.
  bool* evaluate_cost_functions_result;

  // Buffers to hold outputs from evaluation.
  double* costs;
  double* residuals;
  double* gradient;
  double* global_jacobian_values;

  // The jacobians stored in a format where all jacobians
  // for the same residual block are stored contiguously.
  // In the final jacobian matrix, depending on the sparse
  // matrix format, these values may not all be contiguous.
  // These final values are stored in global_jacobian_values
  // and the final locations are determined by
  // jacobian_per_residual_layout and jacobian_per_residual_offsets.
  double* local_jacobian_values;

  // Scratch space to hold intermediate jacobian results
  // prior to potentially  multiplying by a plus jacobian.
  double* jacobian_scratch;

  // Flag that is set to false if the loss function is of
  // type TrivialLossCUDA.
  bool apply_loss_function;

  // Flags to determine which outputs were requested from evaluation.
  bool output_residuals;
  bool output_jacobians;
  bool output_gradient;

  // Stores the locations to write the final residuals.
  int* residual_layout;

  // Stores the locations to write the final jacobian values.
  int* jacobian_per_residual_layout;
  int* jacobian_per_residual_offsets;
};


#if __CUDA_ARCH__ < 600
// built-in atomicAdd for doubles is not supported in architectures < 600.
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__device__ void MatrixMultiplyNaive(const double* A,
                                    const double* B,
                                    double* C,
                                    const int num_row_a,
                                    const int num_col_a,
                                    const int num_col_b) {
  for (int i = 0; i < num_row_a; i++) {
    for (int j = 0; j < num_col_b; j++) {
      C[i *  num_col_b + j] = 0;

      for (int k = 0; k < num_col_a; k++) {
        C[i *  num_col_b + j] +=
            A[i * num_col_a + k] * B[k * num_col_b + j];
      }
    }
  }
}

// Performs a matrix transpose vector multiply with the assumption
// that multiple threads could be writing to the same locations in an
// output vector.
//
// This assumption is needed when computing the gradient, since
// the gradient is calculated from all parameter blocks. Because we
// are parallelizing across residual blocks, a thread with a residual
// block that shares parameters with another residual block in another
// thread would cause multiple threads to attempt to accumulate to the
// same location in the gradient vector. Hence, we need to use atomicAdd().
__device__ void MatrixTransposeVectorMultiplyAtomic(const double* A,
                                                    const int num_row_a,
                                                    const int num_col_a,
                                                    const double* b,
                                                    double* c) {
  for (int i = 0; i < num_col_a; ++i) {
    for (int j = 0; j < num_row_a; ++j) {
      //c[i] += A[j * num_col_a + i] * b[j];
      atomicAdd(&(c[i]), A[j * num_col_a + i] * b[j]);
    }
  }
}

// Uses Eigen
__device__ void MatrixMultiply(const double* A,
                               const double* B,
                               double* C,
                               const int num_row_a,
                               const int num_col_a,
                               const int num_col_b) {
  ConstMatrixRef A_mat(A, num_row_a, num_col_a);
  ConstMatrixRef B_mat(B, num_col_a, num_col_b);
  MatrixRef C_mat(C, num_row_a, num_col_b);
  C_mat.noalias() = A_mat * B_mat;
}

// Uses Eigen (non-atomic)
__device__ void MatrixTransposeVectorMultiply(const double* A,
                                              const int num_row_a,
                                              const int num_col_a,
                                              const double* b,
                                              double* c) {
  ConstMatrixRef A_mat(A, num_row_a, num_col_a);
  ConstVectorRef b_vec(b, num_row_a);
  VectorRef c_vec(c, num_col_a);
  c_vec.noalias() = A_mat.transpose() * b_vec;
}

// Multiplies the transposed final jacobian and residual vector
// together to compute the gradient.
template <typename CostFunctor,
          typename LossFunctionCUDA,
          int kNumResiduals,
          int... Ns>
__device__ void ComputeGradient(ParameterBlockCUDA** parameter_blocks,
                                const EvaluatorCUDAKernelArguments<CostFunctor, LossFunctionCUDA, kNumResiduals, Ns...>& kernel_args,
                                double** block_jacobians,
                                double* block_residuals) {
  using AutoDiffResidualBlockCUDA_t =
      AutoDiffResidualBlockCUDA<CostFunctor, LossFunctionCUDA, kNumResiduals, Ns...>;

  constexpr int num_parameter_blocks = sizeof...(Ns);

  for (int i = 0; i < num_parameter_blocks; ++i) {
    if (parameter_blocks[i]->IsConstant()) {
      continue;
    }

    int tangent_size = parameter_blocks[i]->GetTangentSize();
    int delta_offset = parameter_blocks[i]->GetDeltaOffset();

    double* local_gradient = kernel_args.gradient + delta_offset;
    MatrixTransposeVectorMultiplyAtomic(block_jacobians[i],
                                        kNumResiduals,
                                        tangent_size,
                                        block_residuals,
                                        local_gradient);
  }
}

// Assigns destinations for writing jacobians for a given residual block.
template <typename CostFunctor,
          typename LossFunctionCUDA,
          int kNumResiduals,
          int... Ns>
__device__ void PrepareJacobians(const int thread_id,
                                 const EvaluatorCUDAKernelArguments<CostFunctor, LossFunctionCUDA, kNumResiduals, Ns...>& kernel_args,
                                 ParameterBlockCUDA** parameter_blocks,
                                 const int* parameter_block_sizes,
                                 double** block_jacobians,
                                 double** final_jacobians) {
  using AutoDiffResidualBlockCUDA_t =
      AutoDiffResidualBlockCUDA<CostFunctor, LossFunctionCUDA, kNumResiduals, Ns...>;

  constexpr int num_parameter_blocks = sizeof...(Ns);

  int jacobian_values_offset = 0;
  int jacobian_begin = thread_id *
                       kernel_args.num_parameters_per_residual *
                       kNumResiduals;
  for (int i = 0; i < num_parameter_blocks; i++) {
    double* plus_jacobian = parameter_blocks[i]->GetPlusJacobian();

    final_jacobians[i]
      = &(kernel_args.local_jacobian_values[jacobian_begin + jacobian_values_offset]);

    if (!plus_jacobian) {
      block_jacobians[i]
        = &(kernel_args.local_jacobian_values[jacobian_begin + jacobian_values_offset]);
    } else {
      block_jacobians[i]
        = &(kernel_args.jacobian_scratch[jacobian_begin + jacobian_values_offset]);
    }

    jacobian_values_offset += parameter_block_sizes[i] * kNumResiduals;
  }
}

// Writes the final jacobian values in the correct locations determined
// by the format of the sparse matrix. The format is provided by the data
// structures: jacobian_per_residual_layout, jacobian_per_residual_offsets.
template <typename CostFunctor,
          typename LossFunctionCUDA,
          int kNumResiduals,
          int... Ns>
__device__ void WriteJacobians(const int global_residual_block_index,
                               ParameterBlockCUDA** parameter_blocks,
                               double** local_jacobian_block_ptrs,
                               const EvaluatorCUDAKernelArguments<CostFunctor, LossFunctionCUDA, kNumResiduals, Ns...>& kernel_args) {
  constexpr int num_parameter_blocks = sizeof...(Ns);
  int* jacobian_per_residual_layout = kernel_args.jacobian_per_residual_layout;
  int* jacobian_per_residual_offsets = kernel_args.jacobian_per_residual_offsets;
  double* global_jacobian_values = kernel_args.global_jacobian_values;

  int jacobian_block_index =
      jacobian_per_residual_layout[global_residual_block_index];
  for (int j = 0; j < num_parameter_blocks; ++j) {
    int tangent_size =
        parameter_blocks[j]->GetTangentSize();

    if (!parameter_blocks[j]->IsConstant()) {
      for (int k = 0; k < kNumResiduals; ++k) {
        double* jacobians = global_jacobian_values +
            jacobian_per_residual_offsets[jacobian_block_index];
        int copy_offset = tangent_size * k;
        for (int m = 0; m < tangent_size; ++m) {
          jacobians[m] = local_jacobian_block_ptrs[j][copy_offset + m];
        }

        // Remember that the layout strips out any blocks for inactive
        // parameters. Instead, bump the pointer for active parameters only.
        jacobian_block_index++;
      }
    }
  }
}

// The CUDA kernel that evaluates residual blocks in parallel.
template <typename CostFunctor,
          typename LossFunctionCUDA,
          int kNumResiduals,
          int... Ns>
__global__ void EvaluateKernel(
    EvaluatorCUDAKernelArguments<CostFunctor, LossFunctionCUDA, kNumResiduals, Ns...> kernel_args) {
  using AutoDiffResidualBlockCUDA_t =
      AutoDiffResidualBlockCUDA<CostFunctor, LossFunctionCUDA, kNumResiduals, Ns...>;

  constexpr int num_parameter_blocks = sizeof...(Ns);

  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int num_residual_blocks = kernel_args.num_residual_blocks;
  int parameter_block_sizes[num_parameter_blocks]{Ns...};
  double* block_jacobians[num_parameter_blocks];
  double* final_jacobians[num_parameter_blocks];

  if (thread_id >= num_residual_blocks)
    return;

  // If a cost function evaluation in another thread has
  // already returned false, abort evaluation in all threads.
  if (!(*(kernel_args.evaluate_cost_functions_result)))
    return;

  AutoDiffResidualBlockCUDA_t* residual_block =
      &kernel_args.residual_blocks[thread_id];
  int global_residual_block_index = residual_block->GetGlobalResidualBlockIndex();
  ParameterBlockCUDA** parameter_blocks = residual_block->GetParameterBlocks();

  PrepareJacobians(thread_id,
                   kernel_args,
                   parameter_blocks,
                   parameter_block_sizes,
                   block_jacobians,
                   final_jacobians);

  // Set up parameter block states for residual_block->Evaluate()
  double* parameter_state_ptrs[num_parameter_blocks];
  for (int i = 0; i < num_parameter_blocks; ++i) {
    parameter_state_ptrs[i] = parameter_blocks[i]->GetState();
  }

  double* block_residuals = kernel_args.residuals +
                            kernel_args.residual_layout[global_residual_block_index];
  bool result =
      residual_block->Evaluate(parameter_state_ptrs,
                               block_residuals,
                               block_jacobians);

  if (!result) {
    // The cost function evaluation returned false.
    // Abort this thread and update the global variable
    // to notify all other threads to abort.
    *(kernel_args.evaluate_cost_functions_result) = false;
    return;
  }

  if (kernel_args.output_jacobians || kernel_args.output_gradient) {
    for (int i = 0; i < num_parameter_blocks; i++) {
      int tangent_size = parameter_blocks[i]->GetTangentSize();
      double* plus_jacobian = parameter_blocks[i]->GetPlusJacobian();

      // If the parameter block has a manifold, multiply
      // by its plus jacobian.
      if (plus_jacobian) {
        MatrixMultiply(block_jacobians[i],
                       plus_jacobian,
                       final_jacobians[i],
                       kNumResiduals,
                       parameter_block_sizes[i],
                       tangent_size);
      }
    }
  }

  double squared_norm = VectorRef(block_residuals, kNumResiduals).squaredNorm();
  if (!kernel_args.apply_loss_function) {
    kernel_args.costs[thread_id] = 0.5 * squared_norm;
  } else {
    double rho[3];

    residual_block->EvaluateLossFunction(squared_norm, rho);
    kernel_args.costs[thread_id] = 0.5 * rho[0];

    // Correct the residuals and jacobians with the loss function outputs.
    if (kernel_args.output_residuals ||
        kernel_args.output_jacobians ||
        kernel_args.output_gradient) {
      // Correct residuals and jacobians.
      Corrector corrector(squared_norm, rho);

      if (kernel_args.output_jacobians ||
          kernel_args.output_gradient) {
        for (int i = 0; i < num_parameter_blocks; i++) {
          if (parameter_blocks[i]->IsConstant()) {
            continue;
          }

          corrector.CorrectJacobian(kNumResiduals,
                                    parameter_blocks[i]->GetTangentSize(),
                                    block_residuals,
                                    final_jacobians[i]);
        }
      }

      if (kernel_args.output_residuals) {
        corrector.CorrectResiduals(kNumResiduals, block_residuals);
      }
    }
  }

  if (kernel_args.output_gradient) {
     ComputeGradient(parameter_blocks,
                     kernel_args,
                     final_jacobians,
                     block_residuals);
  }

  if (kernel_args.output_jacobians) {
    WriteJacobians(global_residual_block_index,
                   parameter_blocks,
                   final_jacobians,
                   kernel_args);
  }
}

}  // namespace ceres::internal

#endif /* INTERNAL_CERES_PROGRAM_EVALUATOR_HOST_KERNELS_H_ */
