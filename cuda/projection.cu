// projection.cu

#include "checks.cuh"
#include "gsplat/cuda_forward.hpp"

__global__ void cam_extr_proj_kernel(const float *__restrict__ xyz_w, const float *__restrict__ T, const int N,
                                     float *xyz_c) {
  constexpr int XYZ_STRIDE = 3;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = threadIdx.x & 0x1f; // lane_id in warp (0-31)

  // Load and broadcast Extrinsic Matrix T (3x4) within warp
  float t_val = 0.0f;
  if (lane_id < 12) {
    t_val = T[lane_id];
  }
  // T = [r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2]
  const float t00 = __shfl_sync(0xffffffff, t_val, 0);
  const float t01 = __shfl_sync(0xffffffff, t_val, 1);
  const float t02 = __shfl_sync(0xffffffff, t_val, 2);
  const float t03 = __shfl_sync(0xffffffff, t_val, 3);
  const float t10 = __shfl_sync(0xffffffff, t_val, 4);
  const float t11 = __shfl_sync(0xffffffff, t_val, 5);
  const float t12 = __shfl_sync(0xffffffff, t_val, 6);
  const float t13 = __shfl_sync(0xffffffff, t_val, 7);
  const float t20 = __shfl_sync(0xffffffff, t_val, 8);
  const float t21 = __shfl_sync(0xffffffff, t_val, 9);
  const float t22 = __shfl_sync(0xffffffff, t_val, 10);
  const float t23 = __shfl_sync(0xffffffff, t_val, 11);

  if (i >= N) {
    return;
  }

  // Load world-space point
  const float wx = xyz_w[i * XYZ_STRIDE + 0];
  const float wy = xyz_w[i * XYZ_STRIDE + 1];
  const float wz = xyz_w[i * XYZ_STRIDE + 2];

  // Matrix-vector multiply to get camera-space point xyz_c
  xyz_c[i * XYZ_STRIDE + 0] = t00 * wx + t01 * wy + t02 * wz + t03;
  xyz_c[i * XYZ_STRIDE + 1] = t10 * wx + t11 * wy + t12 * wz + t13;
  xyz_c[i * XYZ_STRIDE + 2] = t20 * wx + t21 * wy + t22 * wz + t23;
}

__global__ void cam_intr_proj_kernel(const float *__restrict__ xyz, const float *__restrict__ K, const int N,
                                     float *uv) {
  constexpr int XYZ_STRIDE = 3;
  constexpr int UV_STRIDE = 2;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = threadIdx.x & 0x1f;

  // load and broadcast K to all threads in warp
  float k_val = 0.0f;
  if (lane_id < 10) {
    k_val = K[lane_id];
  }
  // K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
  const float fx = __shfl_sync(0xffffffff, k_val, 0);
  const float cx = __shfl_sync(0xffffffff, k_val, 2);
  const float fy = __shfl_sync(0xffffffff, k_val, 4);
  const float cy = __shfl_sync(0xffffffff, k_val, 5);

  if (i >= N) {
    return;
  }

  const float x = xyz[i * XYZ_STRIDE + 0];
  const float y = xyz[i * XYZ_STRIDE + 1];
  const float z = xyz[i * XYZ_STRIDE + 2];

  uv[i * UV_STRIDE + 0] = fx * x / z + cx;
  uv[i * UV_STRIDE + 1] = fy * y / z + cy;
}

void camera_extrinsic_projection(float *const xyz_w, const float *T, const int N, float *xyz_c, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(xyz_w);
  ASSERT_DEVICE_POINTER(T);
  ASSERT_DEVICE_POINTER(xyz_c);

  const int threads_per_block = 1024;
  // Calculate the number of blocks needed to cover all N points
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  cam_extr_proj_kernel<<<gridsize, blocksize, 0, stream>>>(xyz_w, T, N, xyz_c);
}

void camera_intrinsic_projection(float *const xyz, const float *K, const int N, float *uv, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(xyz);
  ASSERT_DEVICE_POINTER(K);
  ASSERT_DEVICE_POINTER(uv);

  const int threads_per_block = 1024;
  // Calculate the number of blocks needed to cover all N points
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  cam_intr_proj_kernel<<<gridsize, blocksize, 0, stream>>>(xyz, K, N, uv);
}
