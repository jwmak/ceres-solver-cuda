#ifndef CERES_PUBLIC_INTERNAL_CUDA_DEFS_H_
#define CERES_PUBLIC_INTERNAL_CUDA_DEFS_H_

#include <cmath>

#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA
  #ifdef __CUDACC__ // compiling using nvcc
    #define HOST_DEVICE __host__ __device__
    #define DEVICE __device__
  #else
    #define HOST_DEVICE
    #define DEVICE
  #endif  // __CUDACC__
  #ifdef __CUDA_ARCH__ // compiling device code
    #define DEVICE_CODE
  #endif // __CUDA_ARCH__
#else
  #define HOST_DEVICE
  #define DEVICE
#endif  // CERES_NO_CUDA

#endif  // CERES_PUBLIC_INTERNAL_CUDA_DEFS_H_
