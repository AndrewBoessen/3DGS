// projection_backward.cu

#include "checks.cuh"
#include "gsplat_cuda/cuda_backward.cuh"

__global__ void project_to_screen_backward_kernel(const float *__restrict__ xyz_c, const float *__restrict__ proj,
                                                  const float *__restrict__ uv_grad_out, const int N, const int width,
                                                  const int height, float *__restrict__ xyz_c_grad_in) {
  constexpr int XYZ_STRIDE = 3;
  constexpr int UV_STRIDE = 2;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = threadIdx.x & 0x1f; // lane_id in warp (0-31)

  // Load and broadcast Proj Matrix within warp
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

  const float x = xyz_c[i * XYZ_STRIDE + 0];
  const float y = xyz_c[i * XYZ_STRIDE + 1];
  const float z = xyz_c[i * XYZ_STRIDE + 2];

  // Forward pass recomputation
  float x_clip = p00 * x + p01 * y + p02 * z + p03;
  float y_clip = p10 * x + p11 * y + p12 * z + p13;
  float w_clip = p30 * x + p31 * y + p32 * z + p33;

  // Avoid division by zero
  if (fabsf(w_clip) < 1e-6f) {
    return;
  }

  const float w_inv = 1.0f / w_clip;
  const float w_inv2 = w_inv * w_inv;

  const float grad_u = uv_grad_out[i * UV_STRIDE + 0];
  const float grad_v = uv_grad_out[i * UV_STRIDE + 1];

  // d(NDC) / d(uv)
  float dx_ndc = grad_u * width * 0.5f;
  float dy_ndc = grad_v * height * 0.5f;

  // d(Clip) / d(NDC)
  float dx_clip = dx_ndc * w_inv;
  float dy_clip = dy_ndc * w_inv;
  float dw_clip = -dx_ndc * x_clip * w_inv2 - dy_ndc * y_clip * w_inv2;
  float dz_clip = 0.0f;

  // d(xyz_c) / d(Clip) = Proj^T * d(Clip)
  xyz_c_grad_in[i * XYZ_STRIDE + 0] += p00 * dx_clip + p10 * dy_clip + p20 * dz_clip + p30 * dw_clip;
  xyz_c_grad_in[i * XYZ_STRIDE + 1] += p01 * dx_clip + p11 * dy_clip + p21 * dz_clip + p31 * dw_clip;
  xyz_c_grad_in[i * XYZ_STRIDE + 2] += p02 * dx_clip + p12 * dy_clip + p22 * dz_clip + p32 * dw_clip;
}

void project_to_screen_backward(const float *const xyz_c, const float *const proj, const float *const uv_grad_out,
                                const int N, const int width, const int height, float *xyz_c_grad_in,
                                cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(xyz_c);
  ASSERT_DEVICE_POINTER(proj);
  ASSERT_DEVICE_POINTER(uv_grad_out);
  ASSERT_DEVICE_POINTER(xyz_c_grad_in);

  const int threads_per_block = 256;
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  project_to_screen_backward_kernel<<<gridsize, blocksize, 0, stream>>>(xyz_c, proj, uv_grad_out, N, width, height,
                                                                        xyz_c_grad_in);
}

__global__ void compute_camera_space_points_backward_kernel(const float *__restrict__ xyz_w,
                                                            const float *__restrict__ view,
                                                            const float *__restrict__ xyz_c_grad_out, const int N,
                                                            float *__restrict__ xyz_w_grad_in) {
  constexpr int XYZ_STRIDE = 3;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = threadIdx.x & 0x1f;

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

  const float grad_x_c = xyz_c_grad_out[i * XYZ_STRIDE + 0];
  const float grad_y_c = xyz_c_grad_out[i * XYZ_STRIDE + 1];
  const float grad_z_c = xyz_c_grad_out[i * XYZ_STRIDE + 2];

  // --- Gradient w.r.t. xyz_w ---
  // d(xyz_w) = View^T * d(xyz_c) (ignoring translation part for direction vectors, but xyz_w is point)
  // Actually, d(xyz_w) = R^T * d(xyz_c) because translation is constant w.r.t. xyz_w.
  // The View matrix upper-left 3x3 is the rotation R.
  xyz_w_grad_in[i * XYZ_STRIDE + 0] += v00 * grad_x_c + v10 * grad_y_c + v20 * grad_z_c;
  xyz_w_grad_in[i * XYZ_STRIDE + 1] += v01 * grad_x_c + v11 * grad_y_c + v21 * grad_z_c;
  xyz_w_grad_in[i * XYZ_STRIDE + 2] += v02 * grad_x_c + v12 * grad_y_c + v22 * grad_z_c;
}

void compute_camera_space_points_backward(const float *const xyz_w, const float *const view,
                                          const float *const xyz_c_grad_out, const int N, float *xyz_w_grad_in,
                                          cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(xyz_w);
  ASSERT_DEVICE_POINTER(view);
  ASSERT_DEVICE_POINTER(xyz_c_grad_out);
  ASSERT_DEVICE_POINTER(xyz_w_grad_in);

  const int threads_per_block = 256;
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  compute_camera_space_points_backward_kernel<<<gridsize, blocksize, 0, stream>>>(xyz_w, view, xyz_c_grad_out, N,
                                                                                  xyz_w_grad_in);
}
