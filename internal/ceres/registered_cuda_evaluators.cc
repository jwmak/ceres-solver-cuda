#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA

#include "ceres/internal/registered_cuda_evaluators.h"

#include <algorithm>
#include <iostream>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "ceres/internal/eigen.h"
#include "ceres/internal/parameter_block_cuda.h"
#include "ceres/internal/residual_block_cuda.h"
#include "ceres/internal/residual_block_cuda_evaluator.h"
#include "ceres/parameter_block.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"
#include "glog/logging.h"

namespace ceres::internal {

RegisteredCUDAEvaluators::RegisteredCUDAEvaluators(ContextImpl* context)
    : context_{context},
      d_parameter_blocks_{context},
      d_constant_parameter_blocks_{context},
      d_parameters_state_{context},
      d_constant_parameters_state_{context},
      d_plus_jacobians_{context},
      d_residuals_{context},
      d_residual_layout_{context},
      d_jacobian_values_{context},
      d_jacobian_per_residual_layout_{context},
      d_jacobian_per_residual_offsets_{context},
      d_gradient_{context} {
  if (!context_->IsCudaInitialized()) {
    std::string message;
    if (!context_->InitCuda(&message)) {
      LOG(ERROR) << "Terminating: " << message;
      return;
    }
  }
}

bool RegisteredCUDAEvaluators::Evaluate(const double* state,
                                        double* cost,
                                        double* residuals,
                                        double* gradient,
                                        double* jacobian_values) {
  *cost = 0.0;
  // Update the state of each non-constant parameter block in
  // device memory.
  d_parameters_state_.CopyFromCpu(state, d_parameters_state_.size());
  UpdatePlusJacobians();

  if (residuals != nullptr)
    d_residuals_.Zero();

  double* d_jacobian_values = nullptr;
  if (jacobian_values != nullptr) {
    d_jacobian_values_.Zero();
    d_jacobian_values = d_jacobian_values_.data();
  }

  double* d_gradient = nullptr;
  if (gradient != nullptr) {
    d_gradient_.Zero();
    d_gradient = d_gradient_.data();
  }

  // Iterate through each registered CUDA evaluator and call Evaluate().
  for (auto& [evaluator_type, residual_block_cuda_evaluator] : evaluators_) {
    VLOG(3) << "Evaluating "
            << evaluators_[evaluator_type]->NumResidualBlocks()
            << " residual blocks of type: "
            << evaluator_type.name();;

    bool result = residual_block_cuda_evaluator->Evaluate(
                                            cost,
                                            residuals != nullptr,
                                            d_residuals_.data(),
                                            d_residual_layout_.data(),
                                            d_gradient,
                                            d_jacobian_values,
                                            d_jacobian_per_residual_layout_.data(),
                                            d_jacobian_per_residual_offsets_.data());

    if (!result)
      return false;
  }

  if (residuals != nullptr)
    d_residuals_.CopyToCpu(residuals, d_residuals_.size());

  if (jacobian_values != nullptr)
    d_jacobian_values_.CopyToCpu(jacobian_values, d_jacobian_values_.size());

  if (gradient != nullptr)
    d_gradient_.CopyToCpu(gradient, d_gradient_.size());

  return true;
}

void RegisteredCUDAEvaluators::UpdatePlusJacobians() {
  int plus_jacobian_offset = 0;
  for (int i = 0; i < plus_jacobians_.size(); ++i) {
    const double* plus_jacobian = plus_jacobians_[i];
    int plus_jacobian_size = plus_jacobian_sizes_[i];
    std::copy(plus_jacobian,
              plus_jacobian + plus_jacobian_size,
              h_plus_jacobians_.data() + plus_jacobian_offset);
    plus_jacobian_offset += plus_jacobian_size;
  }

  // Copy plus jacobians array to device memory all at once.
  if (h_plus_jacobians_.size()) {
    d_plus_jacobians_.CopyFromCpu(h_plus_jacobians_.data(),
                                  h_plus_jacobians_.size());
  }
}

void RegisteredCUDAEvaluators::SetupParameterBlocks(Program* program) {
  const std::vector<ParameterBlock*>& parameter_blocks = program->parameter_blocks();
  std::vector<ParameterBlockCUDA> h_parameter_blocks_cuda(parameter_blocks.size());

  plus_jacobians_.clear();
  plus_jacobian_sizes_.clear();
  h_plus_jacobians_.clear();
  int total_plus_jacobian_values = 0;

  // Set up non-constant parameter blocks.
  for (int i = 0; i < parameter_blocks.size(); ++i) {
    ParameterBlock* parameter_block = parameter_blocks[i];
    ParameterBlockCUDA* parameter_block_cuda = &h_parameter_blocks_cuda[i];

    parameter_block_cuda->SetTangentSize(parameter_block->TangentSize());
    parameter_block_cuda->SetDeltaOffset(parameter_block->delta_offset());
    parameter_block_cuda->SetStateOffset(parameter_block->state_offset());
    parameter_block_cuda->SetState(d_parameters_state_.data() +
                                   parameter_block->state_offset());

    const double* plus_jacobian = parameter_block->PlusJacobian();
    if (plus_jacobian) {
      const int tangent_size = parameter_block->TangentSize();
      const int ambient_size = parameter_block->Size();

      // Plus jacobians are stored contigously in a vector, where
      // they will be copied to device memory. Each ParameterBlockCUDA
      // that has a plus jacobian will have a pointer to some location
      // in the device memory array.
      plus_jacobians_.push_back(plus_jacobian);
      plus_jacobian_sizes_.push_back(tangent_size * ambient_size);
      total_plus_jacobian_values += tangent_size * ambient_size;
    }
  }
  h_plus_jacobians_.resize(total_plus_jacobian_values);
  d_plus_jacobians_.Reserve(total_plus_jacobian_values);

  // Set up plus jacobians.
  int plus_jacobian_offset = 0;
  int plus_jacobian_index = 0;
  if (total_plus_jacobian_values > 0) {
    for (int i = 0; i < parameter_blocks.size(); ++i) {
      ParameterBlock* parameter_block = parameter_blocks[i];
      ParameterBlockCUDA* parameter_block_cuda = &h_parameter_blocks_cuda[i];
      const double* plus_jacobian = parameter_block->PlusJacobian();

      if (plus_jacobian) {
        double* d_plus_jacobian =
            d_plus_jacobians_.data() + plus_jacobian_offset;
        parameter_block_cuda->SetPlusJacobian(d_plus_jacobian);
        plus_jacobian_offset += plus_jacobian_sizes_[plus_jacobian_index];
        plus_jacobian_index++;
      }
    }
  }

  const std::vector<ParameterBlock*>& constant_parameter_blocks =
      program->constant_parameter_blocks();
  std::vector<ParameterBlockCUDA> h_constant_parameter_blocks_cuda(constant_parameter_blocks.size());

  // Set up constant parameter blocks.
  for (int i = 0; i < constant_parameter_blocks.size(); ++i) {
    ParameterBlock* parameter_block = constant_parameter_blocks[i];
    ParameterBlockCUDA* parameter_block_cuda = &h_constant_parameter_blocks_cuda[i];

    parameter_block_cuda->SetConstant();
    parameter_block_cuda->SetStateOffset(parameter_block->state_offset());
    parameter_block_cuda->SetState(d_constant_parameters_state_.data() +
                                   parameter_block->state_offset());
  }

  d_parameter_blocks_.CopyFromCpu(h_parameter_blocks_cuda.data(),
                                  h_parameter_blocks_cuda.size());
  d_constant_parameter_blocks_.CopyFromCpu(h_constant_parameter_blocks_cuda.data(),
                                           h_constant_parameter_blocks_cuda.size());
}

void RegisteredCUDAEvaluators::SetupResidualBlocks(Program* program) {
  const std::vector<ResidualBlock*>& residual_blocks = program->residual_blocks();

  for (auto residual_block : residual_blocks) {
    ResidualBlockCUDA* residual_block_cuda = residual_block->GetResidualBlockCUDA();

    // The global residual block index is the position of the residual block
    // in the array containing all of the program's residual blocks.
    residual_block_cuda->SetGlobalResidualBlockIndex(residual_block->index());

    for (int i = 0; i < residual_block->NumParameterBlocks(); ++i) {
      ParameterBlock* parameter_block = residual_block->parameter_blocks()[i];
      int index = parameter_block->index();

      // Constant and non-constant parameter blocks are stored in different buffers.
      if (parameter_block->IsConstant()) {
        residual_block_cuda->SetParameterBlock(i, d_constant_parameter_blocks_.data() + index);
      } else {
        residual_block_cuda->SetParameterBlock(i, d_parameter_blocks_.data() + index);
      }
    }

    AddResidualBlockCUDA(residual_block_cuda);
  }
}

void RegisteredCUDAEvaluators::Init(Program* program,
                                    const std::vector<int>& residual_layout,
                                    const std::vector<int>& jacobian_per_residual_layout,
                                    const std::vector<int>& jacobian_per_residual_offsets,
                                    int num_jacobian_values) {
  const std::vector<ResidualBlock*>& residual_blocks = program->residual_blocks();

  int num_constant_parameters = program->NumConstantParameters();
  int num_residual_blocks = program->NumResidualBlocks();
  int num_residuals = program->NumResiduals();
  int num_effective_parameters = program->NumEffectiveParameters();
  int num_parameters = program->NumParameters();

  std::vector<double> h_constant_parameter_state(num_constant_parameters);

  // Store the states of the constant parameter blocks in a vector.
  // This only needs to be done once, since the states of constant
  // parameter blocks never change.
  program->ConstantParameterBlocksToStateVector(h_constant_parameter_state.data());

  // Some CudaBuffers need to be freed, since they rely on CudaBuffer::size()
  // to know the correct number of bytes to copy. This size must be reset
  // upon initialization and latered assigned by Reserve() or CopyFromCpu().
  d_parameters_state_.Free();
  d_parameters_state_.Reserve(num_parameters);
  d_constant_parameters_state_.CopyFromCpu(h_constant_parameter_state.data(),
                                           num_constant_parameters);

  d_residuals_.Free();
  d_residuals_.Reserve(num_residuals);
  d_residual_layout_.CopyFromCpu(residual_layout.data(), num_residual_blocks);

  d_jacobian_values_.Free();
  d_jacobian_values_.Reserve(num_jacobian_values);
  d_jacobian_per_residual_layout_.CopyFromCpu(jacobian_per_residual_layout.data(),
                                              jacobian_per_residual_layout.size());
  d_jacobian_per_residual_offsets_.CopyFromCpu(jacobian_per_residual_offsets.data(),
                                               jacobian_per_residual_offsets.size());
  d_gradient_.Free();
  d_gradient_.Reserve(num_effective_parameters);

  for (auto& [evaluator_type, residual_block_cuda_evaluator] : evaluators_) {
    residual_block_cuda_evaluator->Reset();
  }

  SetupParameterBlocks(program);
  SetupResidualBlocks(program);

  for (auto& [evaluator_type, residual_block_cuda_evaluator] : evaluators_) {
    if (residual_block_cuda_evaluator->NumResidualBlocks() > 0)
      residual_block_cuda_evaluator->Init();
    else
      evaluators_.erase(evaluator_type);
  }
}

void RegisteredCUDAEvaluators::AddResidualBlockCUDA(ResidualBlockCUDA* residual_block_cuda) {
  std::type_index evaluator_type = std::type_index(typeid(*residual_block_cuda));
  evaluators_[evaluator_type]->AddResidualBlock(residual_block_cuda);
}

ResidualBlockCUDAEvaluator* RegisteredCUDAEvaluators::GetResidualBlockCUDAEvaluator(
      const std::type_index& evaluator_type) {
  if (evaluators_.find(evaluator_type) == evaluators_.end())
     return nullptr;
  return evaluators_[evaluator_type].get();
}

void RegisteredCUDAEvaluators::RegisterCUDAEvaluator(
    const std::type_index& evaluator_type,
    ResidualBlockCUDAEvaluator* residual_block_cuda_evaluator) {
  evaluators_[evaluator_type].reset(residual_block_cuda_evaluator);
}

}  // namespace ceres::internal

#endif  // CERES_NO_CUDA
