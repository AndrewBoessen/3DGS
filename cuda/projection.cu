// projection.cu

#include "checks.cuh"
#include "gsplat_cuda/cuda_forward.cuh"

__global__ void compute_camera_space_points_kernel(const float *__restrict__ xyz_w, const float *__restrict__ view,
                                                   const int N, float *xyz_c) {
  constexpr int XYZ_STRIDE = 3;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = threadIdx.x & 0x1f; // lane_id in warp (0-31)

  // Load and broadcast View Matrix (4x4) within warp
  float v_val = 0.0f;
  if (lane_id < 16) {
    v_val = view[lane_id];
  }

  const float v00 = __shfl_sync(0xffffffff, v_val, 0);
  const float v01 = __shfl_sync(0xffffffff, v_val, 1);
  const float v02 = __shfl_sync(0xffffffff, v_val, 2);
  const float v03 = __shfl_sync(0xffffffff, v_val, 3);
  const float v10 = __shfl_sync(0xffffffff, v_val, 4);
  const float v11 = __shfl_sync(0xffffffff, v_val, 5);
  const float v12 = __shfl_sync(0xffffffff, v_val, 6);
  const float v13 = __shfl_sync(0xffffffff, v_val, 7);
  const float v20 = __shfl_sync(0xffffffff, v_val, 8);
  const float v21 = __shfl_sync(0xffffffff, v_val, 9);
  const float v22 = __shfl_sync(0xffffffff, v_val, 10);
  const float v23 = __shfl_sync(0xffffffff, v_val, 11);

  if (i >= N) {
    return;
  }

  // Load world-space point
  const float wx = xyz_w[i * XYZ_STRIDE + 0];
  const float wy = xyz_w[i * XYZ_STRIDE + 1];
  const float wz = xyz_w[i * XYZ_STRIDE + 2];

  // Matrix-vector multiply to get camera-space point xyz_c
  xyz_c[i * XYZ_STRIDE + 0] = v00 * wx + v01 * wy + v02 * wz + v03;
  xyz_c[i * XYZ_STRIDE + 1] = v10 * wx + v11 * wy + v12 * wz + v13;
  xyz_c[i * XYZ_STRIDE + 2] = v20 * wx + v21 * wy + v22 * wz + v23;
}

__global__ void project_to_screen_kernel(const float *__restrict__ xyz, const float *__restrict__ proj, const int N,
                                         const int width, const int height, float *uv) {
  constexpr int XYZ_STRIDE = 3;
  constexpr int UV_STRIDE = 2;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = threadIdx.x & 0x1f;

  // load and broadcast Proj to all threads in warp
  float p_val = 0.0f;
  if (lane_id < 16) {
    p_val = proj[lane_id];
  }

  const float p00 = __shfl_sync(0xffffffff, p_val, 0);
  const float p01 = __shfl_sync(0xffffffff, p_val, 1);
  const float p02 = __shfl_sync(0xffffffff, p_val, 2);
  const float p03 = __shfl_sync(0xffffffff, p_val, 3);
  const float p10 = __shfl_sync(0xffffffff, p_val, 4);
  const float p11 = __shfl_sync(0xffffffff, p_val, 5);
  const float p12 = __shfl_sync(0xffffffff, p_val, 6);
  const float p13 = __shfl_sync(0xffffffff, p_val, 7);
  const float p20 = __shfl_sync(0xffffffff, p_val, 8);
  const float p21 = __shfl_sync(0xffffffff, p_val, 9);
  const float p22 = __shfl_sync(0xffffffff, p_val, 10);
  const float p23 = __shfl_sync(0xffffffff, p_val, 11);
  const float p30 = __shfl_sync(0xffffffff, p_val, 12);
  const float p31 = __shfl_sync(0xffffffff, p_val, 13);
  const float p32 = __shfl_sync(0xffffffff, p_val, 14);
  const float p33 = __shfl_sync(0xffffffff, p_val, 15);

  if (i >= N) {
    return;
  }

  const float x = xyz[i * XYZ_STRIDE + 0];
  const float y = xyz[i * XYZ_STRIDE + 1];
  const float z = xyz[i * XYZ_STRIDE + 2];

  // Clip space
  float x_clip = p00 * x + p01 * y + p02 * z + p03;
  float y_clip = p10 * x + p11 * y + p12 * z + p13;
  float w_clip = p30 * x + p31 * y + p32 * z + p33;

  // NDC
  float x_ndc = x_clip / (w_clip + 1e-6f);
  float y_ndc = y_clip / (w_clip + 1e-6f);

  // Screen space
  uv[i * UV_STRIDE + 0] = (x_ndc * 0.5f + 0.5f) * width;
  uv[i * UV_STRIDE + 1] = (y_ndc * 0.5f + 0.5f) * height;
}

void compute_camera_space_points(float *const xyz_w, const float *view, const int N, float *xyz_c,
                                 cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(xyz_w);
  ASSERT_DEVICE_POINTER(view);
  ASSERT_DEVICE_POINTER(xyz_c);

  const int threads_per_block = 256;
  // Calculate the number of blocks needed to cover all N points
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  compute_camera_space_points_kernel<<<gridsize, blocksize, 0, stream>>>(xyz_w, view, N, xyz_c);
}

void project_to_screen(float *const xyz, const float *proj, const int N, const int width, const int height, float *uv,
                       cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(xyz);
  ASSERT_DEVICE_POINTER(proj);
  ASSERT_DEVICE_POINTER(uv);

  const int threads_per_block = 256;
  // Calculate the number of blocks needed to cover all N points
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  project_to_screen_kernel<<<gridsize, blocksize, 0, stream>>>(xyz, proj, N, width, height, uv);
}
