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
// Implements some functions in std, including certain math functions,
// that are not yet supported in CUDA. These functions will run in
// device code only.

#ifndef INCLUDE_CERES_INTERNAL_CUDAMATH_CUDA_MATH_H_
#define INCLUDE_CERES_INTERNAL_CUDAMATH_CUDA_MATH_H_

#include <cmath>
#include <cstdint>

#include "ceres/internal/config.h"
#include "ceres/internal/cuda_defs.h"

namespace ceres::internal::cudamath {

#ifndef CERES_NO_CUDA

DEVICE inline double hypot(const double& x,
                           const double& y,
                           const double& z) {
  double px = abs(x);
  double py = abs(y);
  double pz = abs(z);
  double a = ((px < py) ? (py < pz ? pz : py) : (px < pz ? pz : px));

  if (a != 0.0 && a != -0.0) {
    return a * sqrt((px / a) * (px / a)
                + (py / a) * (py / a)
                + (pz / a) * (pz / a));
  } else {
    return 0;
  }
}

/*
 *  fpclassify.c
 *
 *    by Ian Ollmann
 *
 *  Copyright (c) 2007, Apple Inc. All Rights Reserved.
 *
 *  C99 implementation for __fpclassify function (called by FPCLASSIFY macro.)
 */
DEVICE inline int fpclassify(double x) {
  union{ double d; uint64_t u;}u = {x};

  uint32_t  exp = (uint32_t) ( (u.u & 0x7fffffffffffffffULL) >> 52 );

  if( 0 == exp )
  {
    if( u.u & 0x000fffffffffffffULL )
      return FP_SUBNORMAL;

    return FP_ZERO;
  }

  if( 0x7ff == exp )
  {
    if( u.u & 0x000fffffffffffffULL )
      return FP_NAN;

    return FP_INFINITE;
  }

  return FP_NORMAL;
}

#endif  // CERES_NO_CUDA

}  // namespace ceres::internal::cudamath

#endif /* INCLUDE_CERES_INTERNAL_CUDAMATH_CUDA_MATH_H_ */
