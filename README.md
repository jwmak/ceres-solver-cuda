Ceres Solver with GPU Support for Auto-Differentiated Cost Functions
====================================================================

This [code](https://github.com/jwmak/ceres-solver-cuda) modifies Ceres Solver to support the evaluation and auto-differentiation of cost functions
in parallel on the GPU.
The original Ceres Solver can be found at: https://github.com/ceres-solver/ceres-solver 

## Requirements
1. CUDA 11.3 or higher.
2. Eigen 3.4 or higher.

## Limitations
1. Only cost functions of type `AutoDiffCostFunction` are supported.
2. Only `TRUST_REGION` minimizers are supported.
3. For storing the jacobian, only matrices of type `BlockSparseMatrix` and `CompressedRowSparseMatrix` are supported.
4. The number of residuals in a cost function must be static.

## Guide and Example
When compiling your code with Ceres using `nvcc`, use the flag `--expt-relaxed-constexpr`.

The example found in [bundle_adjuster.cu.cc](examples/bundle_adjuster.cu.cc) solves a typical bundle adjustment 
problem and uses CUDA to evaluate the cost functions in parallel on the GPU.

The example uses the cost functor defined in [snavely_reprojection_error.h](examples/snavely_reprojection_error.h). Within the cost functor, 
the macro `HOST_DEVICE` is prepended to some of its methods. This macro expands to
`___host__ __device__`, which are CUDA qualifiers that indicate the code will run
on both the host and the device. Most importantly, these qualifiers are needed on the functions that get called when evaluating your cost functor.
You should ensure that you preprend these qualifiers to all the necessary functions. In general, be aware
that all data and functions in your cost functor need to work on the GPU. As an example, you cannot have a
`std::unique_ptr` as a member variable in your cost functor because CUDA does not currently support
`std::unique_ptr`.

```cpp
struct SnavelyReprojectionError {
  HOST_DEVICE SnavelyReprojectionError(double observed_x, 
                                       double observed_y)
  : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  HOST_DEVICE bool operator()(const T* const camera,
                              const T* const point,
                              T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    AngleAxisRotatePoint(camera, point, p);

    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    const T xp = -p[0] / p[2];
    const T yp = -p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    const T r2 = xp * xp + yp * yp;
    const T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T& focal = camera[6];
    const T predicted_x = focal * distortion * xp;
    const T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
        const double observed_y) {
     return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
              new SnavelyReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};

```

The following shows an example of creating a new Ceres problem and adding a residual block assigned with the cost functor defined above.
The complete example is found in [bundle_adjuster.cu.cc](examples/bundle_adjuster.cu.cc).
Unlike a typical Ceres problem, we use `ceres::ProblemCUDA` instead of `ceres::Problem` to enable
CUDA for parallel cost function evaluation. During evaluation, the computation is parallelized across all the problem's residual blocks,
and each residual block is assigned to a single GPU thread.

In the example, `camera` is an array of 9 doubles holding
the camera parameters and `point` is an array of 3 doubles holding the point parameters. Note that the
cost functor type (in this case `SnavelyReprojectionError`) needs to be specified as a template parameter
along with the number of residuals and the parameter block sizes. For the loss function,
we use `HuberLossCUDA` defined in [loss_function_cuda.h](include/ceres/loss_function_cuda.h).


```cpp
#include "ceres/problem_cuda.h"


CostFunction* cost_function = 
  SnavelyReprojectionError::Create(3.4, 201.8);

ProblemCUDA problem;
problem->AddResidualBlock<SnavelyReprojectionError, 2, 9, 3>(cost_function, 
                                                             new HuberLossCUDA(1.0), 
                                                             camera, 
                                                             point);

```

## Manifolds

You can set a manifold on a parameter block in the typical way when using Ceres Solver, e.g..
```cpp
SubsetManifold* manifold = new SubsetManifold(9, {0});
problem->SetManifold(camera, manifold);
```

## Loss Functions

Like the cost functors, the loss functions must also be compiled for both the host and the device. This means Evaluate()
must be prepended with the qualifiers `__host__ __device__`. All member data and functions 
called by the loss function must be supported in CUDA. 
Some loss functions that work on the GPU are provided in [loss_function_cuda.h](include/ceres/loss_function_cuda.h). 

## Benchmarks

We benchmark two large bundle adjustment problems found in the [BAL dataset](https://grail.cs.washington.edu/projects/bal/): `problem-1778-993923-pre.txt` and `problem-13682-4456117-pre.txt`.
For CPU multi-threading, we use Ceres' existing multi-threading support.

For our CPU, we use the Intel(R) Xeon(R) CPU E5-2630 v3 with 8 cores and support for 16 threads, and for the GPU, we use a NVIDIA Tesla V100.
We compare the benchmarks using 1 CPU thread, 16 CPU threads, and the GPU.

1 CPU thread:
```bash
./bundle_adjuster --input=problem-1778-993923-pre.txt --num_threads=1 --linear_solver=iterative_schur --num_iterations=20
Time (in seconds):
Preprocessor                         6.173792

  Residual only evaluation          19.605268 (20)
  Jacobian & residual evaluation    72.607767 (15)

./bundle_adjuster --input=problem-13682-4456117-pre.txt --num_threads=1 --linear_solver=iterative_schur --num_iterations=20
Time (in seconds):
Preprocessor                        37.747959

  Residual only evaluation         134.315496 (20)
  Jacobian & residual evaluation   334.231880 (11)`
```

16 CPU threads:
```bash
$./bundle_adjuster --input=problem-1778-993923-pre.txt --num_threads=16 --linear_solver=iterative_schur --num_iterations=20
Time (in seconds):
Preprocessor                         6.474194

  Residual only evaluation           2.743571 (20)
  Jacobian & residual evaluation    10.225532 (15)

$./bundle_adjuster --input=problem-13682-4456117-pre.txt --num_threads=16 --linear_solver=iterative_schur --num_iterations=20
Time (in seconds):
Preprocessor                        38.775842

  Residual only evaluation          15.808206 (20)
  Jacobian & residual evaluation    50.007566 (11)
```

GPU:
```bash
$./bundle_adjuster_cuda --input=problem-1778-993923-pre.txt --num_threads=1 --linear_solver=iterative_schur --num_iterations=20
Time (in seconds):
Preprocessor                         7.538003

  Residual only evaluation           0.784664 (20)
  Jacobian & residual evaluation     3.395612 (15)

$./bundle_adjuster_cuda --input=problem-13682-4456117-pre.txt --num_threads=1 --linear_solver=iterative_schur --num_iterations=20
Time (in seconds):
Preprocessor                        46.876302

  Residual only evaluation           3.983222 (20)
  Jacobian & residual evaluation    17.041634 (11)
```

For the preprocessing time, the solver unfortunately incurs overhead to gather all the residual blocks and parameter blocks,
store them in vectors, and copy them to the GPU. This is less of an issue if the solver runs for many iterations. In addition, 
the user could possibly mitigate this cost by combining multiple residual blocks into fewer residual blocks to effectively reduce 
the total number of residual blocks. Another source of overhead occurs when the solver options require the Jacobian to be stored
in the `CompressedSparseRowMatrix` format. In this case, additional time is spent setting up the matrix structure on the GPU.

When using CUDA to evaluate the cost functions for the large bundle adjustment problems, a basic analysis using a V100 GPU
indicates that the primary bottleneck is data transfer between host memory and device memory. Specifically, the biggest issue
is copying the large number of jacobian values from device memory to host memory. 
