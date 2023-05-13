// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2022 Google Inc. All rights reserved.
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
// Check that Jet<T> functions work properly in CUDA device code.
// Jet<T> functions are run on both the host and device and the
// output values are tested for equality within a certain tolerance.

#include "ceres/jet.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cfenv>
#include <cmath>

#include "ceres/internal/config.h"
#include "ceres/internal/cuda_buffer.h"
#include "ceres/internal/cuda_defs.h"
#include "ceres/stringprintf.h"
#include "ceres/test_util.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#ifndef CERES_NO_CUDA
#include <cuda/std/complex>
#endif  // CERES_NO_CUDA

namespace ceres::internal {

namespace {

using J = Jet<double, 2>;

// Convenient shorthand for making a jet.
HOST_DEVICE J MakeJet(double a, double v0, double v1) {
  J z;
  z.a = a;
  z.v[0] = v0;
  z.v[1] = v1;
  return z;
}

double const kTolerance = 1e-13;

bool AreAlmostEqual(double x, double y, double max_abs_relative_difference) {
  if (std::isnan(x) && std::isnan(y)) {
    return true;
  }

  if (std::isinf(x) && std::isinf(y)) {
    return (std::signbit(x) == std::signbit(y));
  }

  double absolute_difference = std::abs(x - y);
  double relative_difference =
      absolute_difference / std::max(std::abs(x), std::abs(y));

  if (std::fpclassify(x) == FP_ZERO || std::fpclassify(y) == FP_ZERO) {
    // If x or y is exactly zero, then relative difference doesn't have any
    // meaning. Take the absolute difference instead.
    relative_difference = absolute_difference;
  }
  return std::islessequal(relative_difference, max_abs_relative_difference);
}

MATCHER_P(IsAlmostEqualTo, y, "") {
  const bool result = (AreAlmostEqual(arg.a, y.a, kTolerance) &&
                       AreAlmostEqual(arg.v[0], y.v[0], kTolerance) &&
                       AreAlmostEqual(arg.v[1], y.v[1], kTolerance));
  if (!result) {
    *result_listener << "\nexpected - actual : " << y - arg;
  }
  return result;
}

}  // namespace

using booltype = uint8_t;

// Pick arbitrary values for x and y.
const J x = MakeJet(2.3, -2.7, 1e-3);
const J y = MakeJet(1.7, 0.5, 1e+2);
const J z = MakeJet(1e-6, 1e-4, 1e-2);

__global__ void CompoundOperatorsKernel(J input1, J input2, J* results) {
  results[0] = input1;
  results[0] += input2;
  results[1] = input1;
  results[1] -= input2;
  results[2] = input1;
  results[2] *= input2;
  results[3] = input1;
  results[3] /= input2;
}

TEST(JetCUDA, CompoundOperators) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 4;
  CudaBuffer<J> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<J> results(num_tests);

  J plus_equals_host = x;
  plus_equals_host += y;
  J minus_equals_host = x;
  minus_equals_host -= y;
  J multiply_equals_host = x;
  multiply_equals_host *= y;
  J divide_equals_host = x;
  divide_equals_host /= y;

  CompoundOperatorsKernel<<<1, 1, 0, context.DefaultStream()>>>(x, y, results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const J& plus_equals_device = results[0];
  const J& minus_equals_device = results[1];
  const J& multiply_equals_device = results[2];
  const J& divide_equals_device = results[3];

  EXPECT_EQ(plus_equals_host, plus_equals_device);
  EXPECT_EQ(minus_equals_host, minus_equals_device);
  EXPECT_EQ(multiply_equals_host, multiply_equals_device);
  EXPECT_EQ(divide_equals_host, divide_equals_device);
}

__global__ void CompoundWithScalarOperatorsKernel(J input1, double scalar, J* results) {
  results[0] = input1;
  results[0] += scalar;
  results[1] = input1;
  results[1] -= scalar;
  results[2] = input1;
  results[2] *= scalar;
  results[3] = input1;
  results[3] /= scalar;
}

TEST(JetCUDA, CompoundWithScalarOperators) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 4;
  CudaBuffer<J> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<J> results(num_tests);

  J plus_equals_scalar_host = x;
  plus_equals_scalar_host += 9.0;
  J minus_equals_scalar_host = x;
  minus_equals_scalar_host -= 9.0;
  J multiply_equals_scalar_host = x;
  multiply_equals_scalar_host *= 9.0;
  J divide_equals_scalar_host = x;
  divide_equals_scalar_host /= 9.0;

  CompoundWithScalarOperatorsKernel<<<1, 1, 0, context.DefaultStream()>>>(x, 9.0, results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const J& plus_equals_scalar_device = results[0];
  const J& minus_equals_scalar_device = results[1];
  const J& multiply_equals_scalar_device = results[2];
  const J& divide_equals_scalar_device = results[3];

  EXPECT_EQ(plus_equals_scalar_host, plus_equals_scalar_device);
  EXPECT_EQ(minus_equals_scalar_host, minus_equals_scalar_device);
  EXPECT_EQ(multiply_equals_scalar_host, multiply_equals_scalar_device);
  EXPECT_EQ(divide_equals_scalar_host, divide_equals_scalar_device);
}

__global__ void UnitaryOperatorsKernel(J input1, J* results) {
  results[0] = +input1;
  results[1] = -input1;
}

TEST(JetCUDA, UnitaryOperators) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 2;
  CudaBuffer<J> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<J> results(num_tests);

  J unitary_plus_host = +x;
  J unitary_minus_host = -x;

  UnitaryOperatorsKernel<<<1, 1, 0, context.DefaultStream()>>>(x, results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const J& unitary_plus_device = results[0];
  const J& unitary_minus_device = results[1];

  EXPECT_EQ(unitary_plus_host, unitary_plus_device);
  EXPECT_EQ(unitary_minus_host, unitary_minus_device);
}

__global__ void BinaryOperatorsKernel(J input1, J input2, J* results) {
  results[0] = input1 + input2;
  results[1] = input1 - input2;
  results[2] = input1 * input2;
  results[3] = input1 / input2;
}

TEST(JetCUDA, BinaryOperators) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 4;
  CudaBuffer<J> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<J> results(num_tests);

  J binary_plus_host = x + y;
  J binary_minus_host = x - y;
  J binary_times_host = x * y;
  J binary_divide_host = x / y;

  BinaryOperatorsKernel<<<1, 1, 0, context.DefaultStream()>>>(x, y, results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const J& binary_plus_device = results[0];
  const J& binary_minus_device = results[1];
  const J& binary_times_device = results[2];
  const J& binary_divide_device = results[3];

  EXPECT_EQ(binary_plus_host, binary_plus_device);
  EXPECT_EQ(binary_minus_host, binary_minus_device);
  EXPECT_EQ(binary_times_host, binary_times_device);
  EXPECT_EQ(binary_divide_host, binary_divide_device);
}

__global__ void BinaryOperatorsWithScalarKernel(J input1, double scalar, J* results) {
  results[0] = input1 + scalar;
  results[1] = scalar + input1;
  results[2] = input1 - scalar;
  results[3] = scalar - input1;
  results[4] = input1 * scalar;
  results[5] = scalar * input1;
  results[6] = input1 / scalar;
  results[7] = scalar / input1;
}

TEST(JetCUDA, BinaryOperatorsWithScalar) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 8;
  CudaBuffer<J> results_device(&context, num_tests);
  std::vector<J> results(num_tests);

  J binary_plus_scalar_host1 = x + 9.0;
  J binary_plus_scalar_host2 = 9.0 + x;
  J binary_minus_scalar_host1 = x - 9.0;
  J binary_minus_scalar_host2 = 9.0 - x;
  J binary_times_scalar_host1 = x * 9.0;
  J binary_times_scalar_host2 = 9.0 * x;
  J binary_divide_scalar_host1 = x / 9.0;
  J binary_divide_scalar_host2 = 9.0 / x;

  BinaryOperatorsWithScalarKernel<<<1, 1, 0, context.DefaultStream()>>>(x, 9.0, results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const J& binary_plus_scalar_device1 = results[0];
  const J& binary_plus_scalar_device2 = results[1];
  const J& binary_minus_scalar_device1 = results[2];
  const J& binary_minus_scalar_device2 = results[3];
  const J& binary_times_scalar_device1 = results[4];
  const J& binary_times_scalar_device2 = results[5];
  const J& binary_divide_scalar_device1 = results[6];
  const J& binary_divide_scalar_device2 = results[7];

  EXPECT_EQ(binary_plus_scalar_host1, binary_plus_scalar_device1);
  EXPECT_EQ(binary_plus_scalar_host2, binary_plus_scalar_device2);
  EXPECT_EQ(binary_minus_scalar_host1, binary_minus_scalar_device1);
  EXPECT_EQ(binary_minus_scalar_host2, binary_minus_scalar_device2);
  EXPECT_EQ(binary_times_scalar_host1, binary_times_scalar_device1);
  EXPECT_EQ(binary_times_scalar_host2, binary_times_scalar_device2);
  EXPECT_EQ(binary_divide_scalar_host1, binary_divide_scalar_device1);
  EXPECT_EQ(binary_divide_scalar_host2, binary_divide_scalar_device2);
}

__global__ void InequalityOperatorsKernel(J input1, J input2, booltype* results) {
  results[0] = input1 < input2;
  results[1] = input2 < input1;
  results[2] = input1 > input2;
  results[3] = input2 > input1;
  results[4] = input1 <= input2;
  results[5] = input2 <= input1;
  results[6] = input1 >= input2;
  results[7] = input2 >= input1;
}

TEST(JetCUDA, InequalityOperators) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 8;
  CudaBuffer<booltype> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<booltype> results(num_tests);

  booltype less_than_host1 = x < y;
  booltype less_than_host2 = y < x;
  booltype not_equals_host1 = x > y;
  booltype not_equals_host2 = y > x;
  booltype less_than_equal_host1 = x <= y;
  booltype less_than_equal_host2 = y <= x;
  booltype not_equals_equal_host1 = x >= y;
  booltype not_equals_equal_host2 = y >= x;

  InequalityOperatorsKernel<<<1, 1, 0, context.DefaultStream()>>>(x, y, results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const booltype& less_than_device1 = results[0];
  const booltype& less_than_device2 = results[1];
  const booltype& not_equals_device1 = results[2];
  const booltype& not_equals_device2 = results[3];
  const booltype& less_than_equal_device1 = results[4];
  const booltype& less_than_equal_device2 = results[5];
  const booltype& not_equals_equal_device1 = results[6];
  const booltype& not_equals_equal_device2 = results[7];

  EXPECT_EQ(less_than_host1, less_than_device1);
  EXPECT_EQ(less_than_host2, less_than_device2);
  EXPECT_EQ(not_equals_host1, not_equals_device1);
  EXPECT_EQ(not_equals_host2, not_equals_device2);
  EXPECT_EQ(less_than_equal_host1, less_than_equal_device1);
  EXPECT_EQ(less_than_equal_host2, less_than_equal_device2);
  EXPECT_EQ(not_equals_equal_host1, not_equals_equal_device1);
  EXPECT_EQ(not_equals_equal_host2, not_equals_equal_device2);;
}

__global__ void InequalityOperatorsWithScalarKernel(J input1, double scalar, booltype* results) {
  results[0] = input1 < scalar;
  results[1] = scalar < input1;
  results[2] = input1 > scalar;
  results[3] = scalar > input1;
  results[4] = input1 <= scalar;
  results[5] = scalar <= input1;
  results[6] = input1 >= scalar;
  results[7] = scalar >= input1;
}

TEST(JetCUDA, InequalityOperatorsWithScalar) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 8;
  CudaBuffer<booltype> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<booltype> results(num_tests);

  booltype less_than_host1 = x < 9.0;
  booltype less_than_host2 = 9.0 < x;
  booltype not_equals_host1 = x > 9.0;
  booltype not_equals_host2 = 9.0 > x;
  booltype less_than_equal_host1 = x <= 9.0;
  booltype less_than_equal_host2 = 9.0 <= x;
  booltype not_equals_equal_host1 = x >= 9.0;
  booltype not_equals_equal_host2 = 9.0 >= x;

  InequalityOperatorsWithScalarKernel<<<1, 1, 0, context.DefaultStream()>>>(x, 9.0, results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const booltype& less_than_device1 = results[0];
  const booltype& less_than_device2 = results[1];
  const booltype& not_equals_device1 = results[2];
  const booltype& not_equals_device2 = results[3];
  const booltype& less_than_equal_device1 = results[4];
  const booltype& less_than_equal_device2 = results[5];
  const booltype& not_equals_equal_device1 = results[6];
  const booltype& not_equals_equal_device2 = results[7];

  EXPECT_EQ(less_than_host1, less_than_device1);
  EXPECT_EQ(less_than_host2, less_than_device2);
  EXPECT_EQ(not_equals_host1, not_equals_device1);
  EXPECT_EQ(not_equals_host2, not_equals_device2);
  EXPECT_EQ(less_than_equal_host1, less_than_equal_device1);
  EXPECT_EQ(less_than_equal_host2, less_than_equal_device2);
  EXPECT_EQ(not_equals_equal_host1, not_equals_equal_device1);
  EXPECT_EQ(not_equals_equal_host2, not_equals_equal_device2);;
}

__global__ void EqualityOperatorsKernel(J input1, J input2, booltype* results) {
  results[0] = input1 == input2;
  results[1] = input2 == input1;
  results[2] = input1 != input2;
  results[3] = input2 != input1;
}

TEST(JetCUDA, EqualityOperators) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 4;
  CudaBuffer<booltype> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<booltype> results(num_tests);

  booltype equals_host1 = x == y;
  booltype equals_host2 = y == x;
  booltype not_equals_host1 = x != y;
  booltype not_equals_host2 = y != x;

  EqualityOperatorsKernel<<<1, 1, 0, context.DefaultStream()>>>(x, y, results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const booltype& equals_device1 = results[0];
  const booltype& equals_device2 = results[1];
  const booltype& not_equals_device1 = results[2];
  const booltype& not_equals_device2 = results[3];

  EXPECT_EQ(equals_host1, equals_device1);
  EXPECT_EQ(equals_host2, equals_device2);
  EXPECT_EQ(not_equals_host1, not_equals_device1);
  EXPECT_EQ(not_equals_host2, not_equals_device2);

  booltype equals_host3 = x == x;
  booltype not_equals_host3 = x != x;

  EqualityOperatorsKernel<<<1, 1, 0, context.DefaultStream()>>>(x, x, results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const booltype& equals_device3 = results[0];
  const booltype& not_equals_device3 = results[2];

  EXPECT_EQ(equals_host3, equals_device3);
  EXPECT_EQ(not_equals_host3, not_equals_device3);
}


__global__ void EqualityOperatorsWithScalarKernel(J input1, double scalar, booltype* results) {
  results[0] = input1 == scalar;
  results[1] = scalar == input1;
  results[2] = input1 != scalar;
  results[3] = scalar != input1;
}

TEST(JetCUDA, EqualityOperatorsWithScalar) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 4;
  CudaBuffer<booltype> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<booltype> results(num_tests);

  booltype equals_host1 = x == 9.0;
  booltype equals_host2 = 9.0 == x;
  booltype not_equals_host1 = x != 9.0;
  booltype not_equals_host2 = 9.0 != x;

  EqualityOperatorsWithScalarKernel<<<1, 1, 0, context.DefaultStream()>>>(x, 9.0, results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const booltype& equals_device1 = results[0];
  const booltype& equals_device2 = results[1];
  const booltype& not_equals_device1 = results[2];
  const booltype& not_equals_device2 = results[3];

  EXPECT_EQ(equals_host1, equals_device1);
  EXPECT_EQ(equals_host2, equals_device2);
  EXPECT_EQ(not_equals_host1, not_equals_device1);
  EXPECT_EQ(not_equals_host2, not_equals_device2);

  // x = MakeJet(2.3, -2.7, 1e-3);
  booltype equals_host3 = x == 2.3;
  booltype equals_host4 = 2.3 == x;
  booltype not_equals_host3 = x != 2.3;
  booltype not_equals_host4 = 2.3 != x;

  EqualityOperatorsWithScalarKernel<<<1, 1, 0, context.DefaultStream()>>>(x, 2.3, results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const booltype& equals_device3 = results[0];
  const booltype& equals_device4 = results[1];
  const booltype& not_equals_device3 = results[2];
  const booltype& not_equals_device4 = results[3];

  EXPECT_EQ(equals_host3, equals_device3);
  EXPECT_EQ(equals_host4, equals_device4);
  EXPECT_EQ(not_equals_host3, not_equals_device3);
  EXPECT_EQ(not_equals_host4, not_equals_device4);
}

__global__ void MathFunctionsKernel(J input1,
                                    J input2,
                                    J input3,
                                    double scalar,
                                    J* results) {
  results[0]  = abs(-input1);
  results[1]  = log(input1);
  results[2]  = log10(input1);
  results[3]  = log1p(input1);
  results[4]  = exp(input1);
  results[5]  = expm1(input1);
  results[6]  = sqrt(input1);
  results[7]  = cos(input1);
  results[8]  = acos(input1);
  results[9]  = sin(input1);
  results[10] = asin(input1);
  results[11] = tan(input1);
  results[12] = atan(input1);
  results[13] = sinh(input1);
  results[14] = cosh(input1);
  results[15] = tanh(input1);
  results[16] = floor(input1);
  results[17] = ceil(input1);
  results[18] = cbrt(input1);
  results[19] = exp2(input1);
  results[20] = log2(input1);
  results[21] = erf(input1);
  results[22] = erfc(input1);
  results[23] = copysign(input1, -input2);
  results[24] = hypot(input1, input2);
  results[25] = hypot(input1, input2, input3);
  results[26] = fma(input1, input2, input3);
  results[27] = atan2(input1, input2);
  results[28] = pow(input1, scalar);
  results[29] = pow(scalar, input1);
  results[30] = pow(input1, input2);
}

TEST(JetCUDA, MathFunctions) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 31;
  CudaBuffer<J> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<J> results(num_tests);

  double scalar = 5.5;
  Jet<std::complex<double>, 1> complex_jet(std::complex(1.5, 3.14), 3);
  J abs_result_host   = abs(-x);
  J log_result_host   = log(x);
  J log10_result_host = log10(x);
  J log1p_result_host = log1p(x);
  J exp_result_host   = exp(x);
  J expm_result_host  = expm1(x);
  J sqrt_result_host  = sqrt(x);
  J cos_result_host   = cos(x);
  J acos_result_host  = acos(x);
  J sin_result_host   = sin(x);
  J asin_result_host  = asin(x);
  J tan_result_host   = tan(x);
  J atan_result_host  = atan(x);
  J sinh_result_host  = sinh(x);
  J cosh_result_host  = cosh(x);
  J tanh_result_host  = tanh(x);
  J floor_result_host = floor(x);
  J ceil_result_host  = ceil(x);
  J cbrt_result_host  = cbrt(x);
  J exp2_result_host  = exp2(x);
  J log2_result_host  = log2(x);
  J erf_result_host   = erf(x);
  J erfc_result_host  = erfc(x);
  J copysign_result_host = copysign(x, -y);
  J hypot2_result_host = hypot(x, y);
  J hypot3_result_host = hypot(x, y, z);
  J fma_result_host = fma(x, y, z);
  J atan2_result_host = atan2(x, y);
  J pow_result_host1 = pow(x, scalar);
  J pow_result_host2 = pow(scalar, x);
  J pow_result_host3 = pow(x, y);

  MathFunctionsKernel<<<1, 1, 0, context.DefaultStream()>>>(x, y, z, scalar, results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const J& abs_result_device   = results[0];
  const J& log_result_device   = results[1];
  const J& log10_result_device = results[2];
  const J& log1p_result_device = results[3];
  const J& exp_result_device   = results[4];
  const J& expm_result_device  = results[5];
  const J& sqrt_result_device  = results[6];
  const J& cos_result_device   = results[7];
  const J& acos_result_device  = results[8];
  const J& sin_result_device   = results[9];
  const J& asin_result_device  = results[10];
  const J& tan_result_device   = results[11];
  const J& atan_result_device  = results[12];
  const J& sinh_result_device  = results[13];
  const J& cosh_result_device  = results[14];
  const J& tanh_result_device  = results[15];
  const J& floor_result_device = results[16];
  const J& ceil_result_device  = results[17];
  const J& cbrt_result_device  = results[18];
  const J& exp2_result_device  = results[19];
  const J& log2_result_device  = results[20];
  const J& erf_result_device   = results[21];
  const J& erfc_result_device  = results[22];
  const J& copysign_result_device = results[23];
  const J& hypot2_result_device = results[24];
  const J& hypot3_result_device = results[25];
  const J& fma_result_device = results[26];
  const J& atan2_result_device = results[27];
  const J& pow_result_device1 = results[28];
  const J& pow_result_device2 = results[29];
  const J& pow_result_device3 = results[30];

  EXPECT_THAT(abs_result_device,   IsAlmostEqualTo(abs_result_host));
  EXPECT_THAT(log_result_device,   IsAlmostEqualTo(log_result_host));
  EXPECT_THAT(log10_result_device, IsAlmostEqualTo(log10_result_host));
  EXPECT_THAT(log1p_result_device, IsAlmostEqualTo(log1p_result_host));
  EXPECT_THAT(exp_result_device,   IsAlmostEqualTo(exp_result_host));
  EXPECT_THAT(expm_result_device,  IsAlmostEqualTo(expm_result_host));
  EXPECT_THAT(sqrt_result_device,  IsAlmostEqualTo(sqrt_result_host));
  EXPECT_THAT(cos_result_device,   IsAlmostEqualTo(cos_result_host));
  EXPECT_THAT(acos_result_device,  IsAlmostEqualTo(acos_result_host));
  EXPECT_THAT(sin_result_device,   IsAlmostEqualTo(sin_result_host));
  EXPECT_THAT(asin_result_device,  IsAlmostEqualTo(asin_result_host));
  EXPECT_THAT(tan_result_device,   IsAlmostEqualTo(tan_result_host));
  EXPECT_THAT(atan_result_device,  IsAlmostEqualTo(atan_result_host));
  EXPECT_THAT(sinh_result_device,  IsAlmostEqualTo(sinh_result_host));
  EXPECT_THAT(cosh_result_device,  IsAlmostEqualTo(cosh_result_host));
  EXPECT_THAT(tanh_result_device,  IsAlmostEqualTo(tanh_result_host));
  EXPECT_THAT(cbrt_result_device,  IsAlmostEqualTo(cbrt_result_host));
  EXPECT_THAT(exp2_result_device,  IsAlmostEqualTo(exp2_result_host));
  EXPECT_THAT(log2_result_device,  IsAlmostEqualTo(log2_result_host));
  EXPECT_THAT(erf_result_device,   IsAlmostEqualTo(erf_result_host));
  EXPECT_THAT(erfc_result_device,  IsAlmostEqualTo(erfc_result_host));
  EXPECT_THAT(hypot2_result_device, IsAlmostEqualTo(hypot2_result_host));
  EXPECT_THAT(hypot3_result_device, IsAlmostEqualTo(hypot3_result_host));
  EXPECT_THAT(fma_result_device, IsAlmostEqualTo(fma_result_host));
  EXPECT_THAT(atan2_result_device, IsAlmostEqualTo(atan2_result_host));
  EXPECT_THAT(pow_result_device1, IsAlmostEqualTo(pow_result_host1));
  EXPECT_THAT(pow_result_device2, IsAlmostEqualTo(pow_result_host2));
  EXPECT_THAT(pow_result_device3, IsAlmostEqualTo(pow_result_host3));

  EXPECT_EQ(floor_result_host, floor_result_device);
  EXPECT_EQ(ceil_result_host, ceil_result_device);
  EXPECT_EQ(copysign_result_host, copysign_result_device);
}

__global__ void ComplexNumbersKernel(double* results) {
  using cuda::std::complex;

  Jet<complex<double>, 2> complex_jet;
  complex_jet.a = complex<double>(1.5, 3.14);
  complex_jet.v[0] = complex<double>(2.1, 4.2);
  complex_jet.v[1] = complex<double>(3.5, 1.8);
  Jet<complex<double>, 2> norm_result = norm(complex_jet);

  results[0] = norm_result.a.real();
  results[1] = norm_result.a.imag();
  results[2] = norm_result.v[0].real();
  results[3] = norm_result.v[0].imag();
  results[4] = norm_result.v[1].real();
  results[5] = norm_result.v[1].imag();
}

TEST(JetCUDA, ComplexNumbers) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 6;
  CudaBuffer<double> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<double> results(num_tests);

  Jet<std::complex<double>, 2> complex_jet;
  complex_jet.a = std::complex<double>(1.5, 3.14);
  complex_jet.v[0] = std::complex<double>(2.1, 4.2);
  complex_jet.v[1] = std::complex<double>(3.5, 1.8);
  Jet<std::complex<double>, 2> norm_result = norm(complex_jet);

  double result_a_real_host = norm_result.a.real();
  double result_a_imag_host = norm_result.a.imag();
  double result_v0_real_host = norm_result.v[0].real();
  double result_v0_imag_host = norm_result.v[0].imag();
  double result_v1_real_host = norm_result.v[1].real();
  double result_v1_imag_host = norm_result.v[1].imag();

  ComplexNumbersKernel<<<1, 1, 0, context.DefaultStream()>>>(results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  double result_a_real_device = results[0];
  double result_a_imag_device = results[1];
  double result_v0_real_device  = results[2];
  double result_v0_imag_device  = results[3];
  double result_v1_real_device = results[4];
  double result_v1_imag_device  = results[5];

  EXPECT_NEAR(result_a_real_device, result_a_real_host, kTolerance);
  EXPECT_NEAR(result_a_imag_device, result_a_imag_host, kTolerance);
  EXPECT_NEAR(result_v0_real_device, result_v0_real_host, kTolerance);
  EXPECT_NEAR(result_v0_imag_device, result_v0_imag_host, kTolerance);
  EXPECT_NEAR(result_v1_real_device, result_v1_real_host, kTolerance);
  EXPECT_NEAR(result_v1_imag_device, result_v1_imag_host, kTolerance);
}

__global__ void MinMaxDimKernel(J input1, J input2, J* results) {
  results[0] = fmin(input1, input1);
  results[1] = fmin(input1, input2);
  results[2] = fmax(input1, input1);
  results[3] = fmax(input1, input2);
  results[4] = fdim(input1, input1);
  results[5] = fdim(input1, input2);
}

TEST(JetCUDA, MinMaxDim) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 6;
  CudaBuffer<J> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<J> results(num_tests);

  J fmin_result_host1 = fmin(x, x);
  J fmin_result_host2 = fmin(x, y);
  J fmax_result_host1 = fmax(x, x);
  J fmax_result_host2 = fmax(x, y);
  J fdim_result_host1 = fdim(x, x);
  J fdim_result_host2 = fdim(x, y);

  MinMaxDimKernel<<<1, 1, 0, context.DefaultStream()>>>(x, y, results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const J& fmin_result_device1 = results[0];
  const J& fmin_result_device2 = results[1];
  const J& fmax_result_device1 = results[2];
  const J& fmax_result_device2 = results[3];
  const J& fdim_result_device1 = results[4];
  const J& fdim_result_device2 = results[5];

  EXPECT_EQ(fmin_result_host1, fmin_result_device1);
  EXPECT_EQ(fmin_result_host2, fmin_result_device2);
  EXPECT_EQ(fmax_result_host1, fmax_result_device1);
  EXPECT_EQ(fmax_result_host2, fmax_result_device2);
  EXPECT_EQ(fdim_result_host1, fdim_result_device1);
  EXPECT_EQ(fdim_result_host2, fdim_result_device2);
}

__global__ void MinMaxDimWithScalarKernel(J input1, double scalar, J* results) {
  results[0] = fmin(input1, scalar);
  results[1] = fmin(input1.a, input1);
  results[2] = fmax(input1, scalar);
  results[3] = fmax(input1.a, input1);
  results[4] = fdim(input1, scalar);
  results[5] = fdim(input1.a, input1);
}

TEST(JetCUDA, MinMaxDimWithScalar) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 6;
  CudaBuffer<J> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<J> results(num_tests);

  J fmin_result_host1 = fmin(x, 9.0);
  J fmin_result_host2 = fmin(x.a, x);
  J fmax_result_host1 = fmax(x, 9.0);
  J fmax_result_host2 = fmax(x.a, x);
  J fdim_result_host1 = fdim(x, 9.0);
  J fdim_result_host2 = fdim(x.a, x);

  MinMaxDimWithScalarKernel<<<1, 1, 0, context.DefaultStream()>>>(x, 9.0, results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const J& fmin_result_device1 = results[0];
  const J& fmin_result_device2 = results[1];
  const J& fmax_result_device1 = results[2];
  const J& fmax_result_device2 = results[3];
  const J& fdim_result_device1 = results[4];
  const J& fdim_result_device2 = results[5];

  EXPECT_EQ(fmin_result_host1, fmin_result_device1);
  EXPECT_EQ(fmin_result_host2, fmin_result_device2);
  EXPECT_EQ(fmax_result_host1, fmax_result_device1);
  EXPECT_EQ(fmax_result_host2, fmax_result_device2);
  EXPECT_EQ(fdim_result_host1, fdim_result_device1);
  EXPECT_EQ(fdim_result_host2, fdim_result_device2);
}

__global__ void BesselKernel(J input1, int n, J* results) {
  results[0] = BesselJ0(input1);
  results[1] = BesselJ1(input1);
  results[2] = BesselJn(n, input1);
}

TEST(JetCUDA, Bessel) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 3;
  CudaBuffer<J> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<J> results(num_tests);

  J BesselJ0_result_host = BesselJ0(x);;
  J BesselJ1_result_host = BesselJ1(x);
  J BesselJn_result_host = BesselJn(5, x);

  BesselKernel<<<1, 1, 0, context.DefaultStream()>>>(x, 5, results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const J& BesselJ0_result_device = results[0];
  const J& BesselJ1_result_device = results[1];
  const J& BesselJn_result_device = results[2];

  EXPECT_THAT(BesselJ0_result_device, IsAlmostEqualTo(BesselJ0_result_host));
  EXPECT_THAT(BesselJ1_result_device, IsAlmostEqualTo(BesselJ1_result_host));
  EXPECT_THAT(BesselJn_result_device, IsAlmostEqualTo(BesselJn_result_host));
}

__global__ void FPFunctionsKernel(booltype* results) {
  J finite_j = MakeJet(3.0, 0, 0);
  J infinite_j = MakeJet(std::numeric_limits<double>::infinity(), 0, 0);
  J nan_j = MakeJet(std::numeric_limits<double>::quiet_NaN(), 0, 0);

  results[0] = isfinite(finite_j);
  results[1] = isfinite(infinite_j);
  results[2] = isinf(infinite_j);
  results[3] = isinf(finite_j);
  results[4] = isnan(nan_j);
  results[5] = isnan(finite_j);
  results[6] = isnormal(finite_j);
  results[7] = isnormal(nan_j);
  results[8] = signbit(-finite_j);
  results[9] = signbit(finite_j);
}

TEST(JetCUDA, FPFunctions) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 10;
  CudaBuffer<booltype> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<booltype> results(num_tests);

  FPFunctionsKernel<<<1, 1, 0, context.DefaultStream()>>>(results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const booltype& isfinite_true = results[0];
  const booltype& isfinite_false = results[1];
  const booltype& isinf_true = results[2];
  const booltype& isinf_false = results[3];
  const booltype& isnan_true = results[4];
  const booltype& isnan_false = results[5];
  const booltype& isnormal_true = results[6];
  const booltype& isnormal_false = results[7];
  const booltype& signbit_true = results[8];
  const booltype& signbit_false = results[9];

  EXPECT_TRUE(isfinite_true);
  EXPECT_FALSE(isfinite_false);
  EXPECT_TRUE(isinf_true);
  EXPECT_FALSE(isinf_false);
  EXPECT_TRUE(isnan_true);
  EXPECT_FALSE(isnan_false);
  EXPECT_TRUE(isnormal_true);
  EXPECT_FALSE(isnormal_false);
  EXPECT_TRUE(signbit_true);
  EXPECT_FALSE(signbit_false);
}

__global__ void FPClassifyKernel(int* results) {
  J zero_j = MakeJet(0, 0, 0);
  J subnormal_j = MakeJet(std::numeric_limits<double>::denorm_min(), 0, 0);
  J normal_j = MakeJet(3.0, 0, 0);
  J infinite_j = MakeJet(std::numeric_limits<double>::infinity(), 0, 0);
  J nan_j = MakeJet(std::numeric_limits<double>::quiet_NaN(), 0, 0);

  results[0] = fpclassify(zero_j);
  results[1] = fpclassify(subnormal_j);
  results[2] = fpclassify(normal_j);
  results[3] = fpclassify(infinite_j);
  results[4] = fpclassify(nan_j);
}

TEST(JetCUDA, FPClassify) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 10;
  CudaBuffer<int> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<int> results(num_tests);

  FPClassifyKernel<<<1, 1, 0, context.DefaultStream()>>>(results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  int classified_zero = results[0];
  int classified_subnormal = results[1];
  int classified_normal = results[2];
  int classified_infinite = results[3];
  int classified_nan = results[4];

  EXPECT_TRUE(classified_zero == FP_ZERO);
  EXPECT_TRUE(classified_subnormal == FP_SUBNORMAL);
  EXPECT_TRUE(classified_normal == FP_NORMAL);
  EXPECT_TRUE(classified_infinite == FP_INFINITE);
  EXPECT_TRUE(classified_nan == FP_NAN);
}

__global__ void ComparisonFunctionsKernel(booltype* results) {
  J a = MakeJet(3.0, 0, 0);
  J b = MakeJet(4.0, 0, 0);
  J nan_j = MakeJet(std::numeric_limits<double>::quiet_NaN(), 0, 0);

  results[0] = isless(a, b);
  results[1] = isless(b, a);
  results[2] = isless(a, 5.0);
  results[3] = isless(5.0, a);

  results[4] = isgreater(a, b);
  results[5] = isgreater(b, a);
  results[6] = isgreater(a, 5.0);
  results[7] = isgreater(5.0, a);

  results[8] = islessequal(a, a);
  results[9] = islessequal(a, b);
  results[10] = islessequal(b, a);
  results[11] = islessequal(a, 5.0);
  results[12] = islessequal(5.0, a);

  results[13] = isgreaterequal(a, a);
  results[14] = isgreaterequal(a, b);
  results[15] = isgreaterequal(b, a);
  results[16] = isgreaterequal(a, 5.0);
  results[17] = isgreaterequal(5.0, a);

  results[18] = islessgreater(a, a);
  results[19] = islessgreater(a, b);
  results[20] = islessgreater(b, a);
  results[21] = islessgreater(a, 5.0);
  results[22] = islessgreater(5.0, a);

  results[23] = isunordered(a, b);
  results[24] = isunordered(a, nan_j);
  results[25] = isunordered(nan_j, a);
}

TEST(JetCUDA, ComparisonFunctions) {
  ContextImpl context;
  std::string cuda_error;
  EXPECT_TRUE(context.InitCuda(&cuda_error)) << cuda_error;

  const int num_tests = 26;
  CudaBuffer<booltype> results_device(&context);
  results_device.Reserve(num_tests);
  std::vector<booltype> results(num_tests);

  ComparisonFunctionsKernel<<<1, 1, 0, context.DefaultStream()>>>(results_device.data());
  results_device.CopyToCpu(results.data(), num_tests);

  const booltype& isless_true1 = results[0];
  const booltype& isless_false1 = results[1];
  const booltype& isless_true2 = results[2];
  const booltype& isless_false2 = results[3];

  const booltype& isgreater_false1 = results[4];
  const booltype& isgreater_true1 = results[5];
  const booltype& isgreater_false2 = results[6];
  const booltype& isgreater_true2 = results[7];

  const booltype& islessequal_true1 = results[8];
  const booltype& islessequal_true2 = results[9];
  const booltype& islessequal_false1 = results[10];
  const booltype& islessequal_true3 = results[11];
  const booltype& islessequal_false2 = results[12];

  const booltype& isgreaterequal_true1 = results[13];
  const booltype& isgreaterequal_false1 = results[14];
  const booltype& isgreaterequal_true2 = results[15];
  const booltype& isgreaterequal_false2 = results[16];
  const booltype& isgreaterequal_true3 = results[17];

  const booltype& islessgreater_false1 = results[18];
  const booltype& islessgreater_true1 = results[19];
  const booltype& islessgreater_true2 = results[20];
  const booltype& islessgreater_true3 = results[21];
  const booltype& islessgreater_true4 = results[22];

  const booltype& isunordered_false1 = results[23];
  const booltype& isunordered_true1 = results[24];
  const booltype& isunordered_true2 = results[25];

  EXPECT_TRUE(isless_true1);
  EXPECT_FALSE(isless_false1);
  EXPECT_TRUE(isless_true2);
  EXPECT_FALSE(isless_false2);
  EXPECT_FALSE(isgreater_false1);
  EXPECT_TRUE(isgreater_true1);
  EXPECT_FALSE(isgreater_false2);
  EXPECT_TRUE(isgreater_true2);
  EXPECT_TRUE(islessequal_true1);
  EXPECT_TRUE(islessequal_true2);
  EXPECT_FALSE(islessequal_false1);
  EXPECT_TRUE(islessequal_true3);
  EXPECT_FALSE(islessequal_false2);
  EXPECT_TRUE(isgreaterequal_true1);
  EXPECT_FALSE(isgreaterequal_false1);
  EXPECT_TRUE(isgreaterequal_true2);
  EXPECT_FALSE(isgreaterequal_false2);
  EXPECT_TRUE(isgreaterequal_true3);
  EXPECT_FALSE(islessgreater_false1);
  EXPECT_TRUE(islessgreater_true1);
  EXPECT_TRUE(islessgreater_true2);
  EXPECT_TRUE(islessgreater_true3);
  EXPECT_TRUE(islessgreater_true4);
  EXPECT_FALSE(isunordered_false1);
  EXPECT_TRUE(isunordered_true1);
  EXPECT_TRUE(isunordered_true2);
}

}  // namespace ceres::internal
