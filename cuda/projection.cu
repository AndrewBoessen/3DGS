#include "checks.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief CUDA kernel to project 3D points to 2D image coordinates using a camera intrinsic matrix.
 * @param[in]  xyz  A pointer to the input array of 3D points (in camera coordinates).
 * @param[in]  K    A pointer to the 3x3 camera intrinsic matrix.
 * @param[in]  N    The total number of 3D points to project.
 * @param[out] uv   A pointer to the output array where the 2D projected coordinates will be stored.
 */
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

/**
 * @brief Launches the CUDA kernel for projecting 3D points to 2D image coordinates.
 * @param[in]  xyz  A device pointer to the input array of 3D points.
 * @param[in]  K    A device pointer to the camera intrinsic matrix.
 * @param[in]  N    The total number of points.
 * @param[out] uv   A device pointer to the output array for 2D coordinates.
 */
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
