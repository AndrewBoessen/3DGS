#include "checks.cuh"
#include "gsplat/cuda_functions.hpp"

__global__ void cam_intr_proj_kernel(const float *__restrict__ xyz, const float *__restrict__ K, const int N,
                                     float *uv) {
  constexpr int XYZ_STRIDE = 3;
  constexpr int UV_STRIDE = 2;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    const float x = xyz[i * XYZ_STRIDE + 0];
    const float y = xyz[i * XYZ_STRIDE + 1];
    const float z = xyz[i * XYZ_STRIDE + 2];

    // K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    const float fx = K[0];
    const float cx = K[2];
    const float fy = K[4];
    const float cy = K[5];

    uv[i * UV_STRIDE + 0] = fx * x / z + cx;
    uv[i * UV_STRIDE + 1] = fy * y / z + cy;
  }
}

void camera_intrinsic_projection(float *const xyz, const float *K, const int N, float *uv) {
  ASSERT_DEVICE_POINTER(xyz);
  ASSERT_DEVICE_POINTER(K);
  ASSERT_DEVICE_POINTER(uv);

  const int threads_per_block = 1024;
  // Calculate the number of blocks needed to cover all N points
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  cam_intr_proj_kernel<<<gridsize, blocksize>>>(xyz, K, N, uv);
}
