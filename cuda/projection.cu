// projection.cu

#include "checks.cuh"
#include "gsplat/cuda_forward.hpp"

__global__ void cam_intr_proj_kernel(const float *__restrict__ xyz, const float *__restrict__ K, const int N,
                                     float *uv) {
  constexpr int XYZ_STRIDE = 3;
  constexpr int UV_STRIDE = 2;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = threadIdx.x & 0x1f;

  // load and broadcast K to all threads in warp
  float k_val = 0.0f;
  if (lane_id < 9) {
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

void camera_extrinsic_projection(float *const xyz_w, const float *T, const int N, float *xyz_c) {
  ASSERT_DEVICE_POINTER(xyz_w);
  ASSERT_DEVICE_POINTER(T);
  ASSERT_DEVICE_POINTER(xyz_c);

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // A is T (3x4), B is xyz_w (4x1), C is xyz_c (3x1)
  const int m = 3; // Rows of T and xyz_c
  const int n = 1; // Columns of xyz_w and xyz_c
  const int k = 4; // Columns of T and rows of xyz_w

  CHECK_CUBLAS(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, T, m, 0, xyz_w, k, k, &beta,
                                         xyz_c, m, m, N));
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
