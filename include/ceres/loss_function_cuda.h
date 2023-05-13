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
// Classes that define loss functions similar to those in loss_function.h,
// but these are intended to be used in CUDA device code. Since classes
// instantiated in host code cannot call virtual functions in device code,
// these classes do not rely on polymorphism, but simply share a common
// Evaluate() function to compute the loss.

#ifndef CERES_PUBLIC_LOSS_FUNCTION_CUDA_H_
#define CERES_PUBLIC_LOSS_FUNCTION_CUDA_H_

#include "ceres/internal/cuda_defs.h"
#include "ceres/internal/disable_warnings.h"
#include "ceres/types.h"

namespace ceres {

// This class is intended to serve as a common base class
// to enable different LossFunctionCUDA to be stored in
// the same container. This makes it easier to track
// the loss function pointers and delete them. Other than
// a pure virtual destructor, no functions are intended to
// be overriden by derived classes.
class CERES_EXPORT LossFunctionCUDABase {
 public:
  // Use a pure virtual destructor to prevent direct
  // instantiation of this class. The destructor is only
  // defined on the host and should never be called in CUDA code.
  virtual ~LossFunctionCUDABase() = 0;
};

LossFunctionCUDABase::~LossFunctionCUDABase() {}

class CERES_EXPORT TrivialLossCUDA : public LossFunctionCUDABase {
 public:
  HOST_DEVICE void Evaluate(double s, double rho[3]) const {
    rho[0] = s;
    rho[1] = 1.0;
    rho[2] = 0.0;
  }
};

class CERES_EXPORT HuberLossCUDA : public LossFunctionCUDABase {
 public:
  HOST_DEVICE explicit HuberLossCUDA(double a) : a_(a), b_(a * a) {}
  HOST_DEVICE void Evaluate(double s, double rho[3]) const {
    if (s > b_) {
      // Outlier region.
      // 'r' is always positive.
      const double r = sqrt(s);
      rho[0] = 2.0 * a_ * r - b_;
      rho[1] = std::max(std::numeric_limits<double>::min(), a_ / r);
      rho[2] = -rho[1] / (2.0 * s);
    } else {
      // Inlier region.
      rho[0] = s;
      rho[1] = 1.0;
      rho[2] = 0.0;
    }
  }

 private:
  const double a_;
  // b = a^2.
  const double b_;
};

class CERES_EXPORT CauchyLossCUDA : public LossFunctionCUDABase {
 public:
  HOST_DEVICE explicit CauchyLossCUDA(double a) : b_(a * a), c_(1 / b_) {}
  HOST_DEVICE void Evaluate(double s, double rho[3]) const {
    const double sum = 1.0 + s * c_;
    const double inv = 1.0 / sum;
    // 'sum' and 'inv' are always positive, assuming that 's' is.
    rho[0] = b_ * log(sum);
    rho[1] = std::max(std::numeric_limits<double>::min(), inv);
    rho[2] = -c_ * (inv * inv);
  }

 private:
  // b = a^2.
  const double b_;
  // c = 1 / a^2.
  const double c_;
};

template <typename LossFunctionCUDA>
class CERES_EXPORT ScaledLossCUDA : public LossFunctionCUDABase {
 public:
  HOST_DEVICE ScaledLossCUDA(const LossFunctionCUDA& rho, double a)
      : rho_(rho), a_(a) {}
  ScaledLossCUDA(const ScaledLossCUDA&) = delete;
  void operator=(const ScaledLossCUDA&) = delete;
  HOST_DEVICE void Evaluate(double s, double rho[3]) const {
    rho_.Evaluate(s, rho);
    rho[0] *= a_;
    rho[1] *= a_;
    rho[2] *= a_;
  }

 private:
  LossFunctionCUDA rho_;
  const double a_;
};

template <>
class CERES_EXPORT ScaledLossCUDA<TrivialLossCUDA> : public LossFunctionCUDABase {
 public:
  HOST_DEVICE ScaledLossCUDA(const TrivialLossCUDA& rho, double a)
      : rho_(rho), a_(a) {}
  ScaledLossCUDA(const ScaledLossCUDA&) = delete;
  void operator=(const ScaledLossCUDA&) = delete;
  HOST_DEVICE void Evaluate(double s, double rho[3]) const {
    rho[0] = a_ * s;
    rho[1] = a_;
    rho[2] = 0.0;
  }

 private:
  TrivialLossCUDA rho_;
  const double a_;
};

}  // namespace ceres

#include "ceres/internal/reenable_warnings.h"

#endif  // CERES_PUBLIC_LOSS_FUNCTION_CUDA_H_
