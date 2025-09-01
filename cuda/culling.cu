#include "checks.cuh"
#include "gsplat/cuda_forward.hpp"

__device__ __forceinline__ bool z_distance_culling(const float z, const float near_thresh, const float far_thresh) {
  return z >= near_thresh && z <= far_thresh;
}
__device__ __forceinline__ bool frustum_culling(const float u, const float v, const int padding, const int width,
                                                const int height) {
  return u >= (-1 * padding) && u <= width && v >= (-1 * padding) && v <= height;
}

__global__ void frustum_culling_kernel(const float *__restrict__ uv, const float *__restrict__ xyz, const int N,
                                       const float near_thresh, const float far_thresh, const int padding,
                                       const int width, const int height, bool *mask) {
  constexpr int XYZ_STRIDE = 3;
  constexpr int UV_STRIDE = 2;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= N) {
    return;
  }

  const float u = uv[i * UV_STRIDE + 0];
  const float v = uv[i * UV_STRIDE + 1];

  const float z = xyz[i * XYZ_STRIDE + 2];

  mask[i] = !(z_distance_culling(z, near_thresh, far_thresh) || frustum_culling(u, v, padding, width, height));
}

void cull_gaussians(float *const uv, float *const xyz, const int N, const float near_thresh, const float far_thresh,
                    const int padding, const int width, const int height, bool *mask) {
  ASSERT_DEVICE_POINTER(uv);
  ASSERT_DEVICE_POINTER(xyz);
  ASSERT_DEVICE_POINTER(mask);

  const int threads_per_block = 1024;
  // Calculate the number of blocks needed to cover all N points
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  frustum_culling_kernel<<<gridsize, blocksize>>>(uv, xyz, N, near_thresh, far_thresh, padding, width, height, mask);
}
