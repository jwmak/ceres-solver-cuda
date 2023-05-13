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
// This class holds the user-passed cost functor that is meant to be
// executed on the GPU. The class is similar to AutoDiffCostFunction
// but is meant to exist in device code. As a result, it contains no
// virtual functions but calls similar functions to those called by
// AutoDiffCostFunction.

#ifndef INTERNAL_CERES_AUTODIFF_COST_FUNCTION_CUDA_H_
#define INTERNAL_CERES_AUTODIFF_COST_FUNCTION_CUDA_H_

#include "ceres/internal/autodiff.h"
#include "ceres/internal/cuda_defs.h"
#include "ceres/internal/parameter_dims.h"
#include "ceres/internal/variadic_evaluate.h"

namespace ceres {

template <typename CostFunctor,
          int kNumResiduals,  // Number of residuals
          int... Ns>          // Number of parameters in each parameter block.
class AutoDiffCostFunctionCUDA {
 public:
  HOST_DEVICE AutoDiffCostFunctionCUDA(const CostFunctor& functor)
  : functor_(functor) {}

  HOST_DEVICE bool Evaluate(double const* const* parameters,
                            double* residuals,
                            double** jacobians) const {
    using ParameterDims = ceres::internal::StaticParameterDims<Ns...>;

    if (!jacobians) {
      return internal::VariadicEvaluate<ParameterDims>(
            functor_, parameters, residuals);
    }

    return internal::AutoDifferentiate<kNumResiduals, ParameterDims>(
          functor_,
          parameters,
          kNumResiduals,
          residuals,
          jacobians);
  }

  void SetFunctor(const CostFunctor& funct) {
    functor_ = funct;
  }

 private:
  CostFunctor functor_;
};

}  // namespace ceres

#endif /* INTERNAL_CERES_AUTODIFF_COST_FUNCTION_CUDA_H_ */
