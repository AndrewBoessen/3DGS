// projection_backward.cu

#include "checks.cuh"
#include "gsplat/cuda_backward.cuh"

__global__ void cam_intr_proj_backward_kernel(const float *__restrict__ xyz_c, const float *__restrict__ K,
                                              const float *__restrict__ uv_grad_out, const int N,
                                              float *__restrict__ xyz_c_grad_in) {
  constexpr int XYZ_STRIDE = 3;
  constexpr int UV_STRIDE = 2;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = threadIdx.x & 0x1f; // lane_id in warp (0-31)

  // Load and broadcast Intrinsic Matrix K within warp
  // K = [fx, 0, cx, 0, fy, cy, 0, 0, 1] stored as [fx, cx, fy, cy]
  float k_val = 0.0f;
  if (lane_id < 9) {
    k_val = K[lane_id];
  }
  const float fx = __shfl_sync(0xffffffff, k_val, 0);
  const float cx = __shfl_sync(0xffffffff, k_val, 2);
  const float fy = __shfl_sync(0xffffffff, k_val, 4);
  const float cy = __shfl_sync(0xffffffff, k_val, 5);

  if (i >= N) {
    return;
  }

  const float x = xyz_c[i * XYZ_STRIDE + 0];
  const float y = xyz_c[i * XYZ_STRIDE + 1];
  const float z = xyz_c[i * XYZ_STRIDE + 2];

  // Avoid division by zero or negative depth
  if (z <= 1e-6f) {
    xyz_c_grad_in[i * XYZ_STRIDE + 0] += 0.0f;
    xyz_c_grad_in[i * XYZ_STRIDE + 1] += 0.0f;
    xyz_c_grad_in[i * XYZ_STRIDE + 2] += 0.0f;
    return;
  }

  const float z_inv = 1.0f / z;
  const float z_inv2 = z_inv * z_inv;

  const float grad_u = uv_grad_out[i * UV_STRIDE + 0];
  const float grad_v = uv_grad_out[i * UV_STRIDE + 1];

  // --- Gradient w.r.t. xyz_c ---
  // du/dx = fx/z, dv/dy = fy/z
  // du/dz = -fx*x/z^2, dv/dz = -fy*y/z^2
  xyz_c_grad_in[i * XYZ_STRIDE + 0] += grad_u * fx * z_inv;
  xyz_c_grad_in[i * XYZ_STRIDE + 1] += grad_v * fy * z_inv;
  xyz_c_grad_in[i * XYZ_STRIDE + 2] += -(grad_u * fx * x * z_inv2 + grad_v * fy * y * z_inv2);
}

void camera_intrinsic_projection_backward(const float *const xyz_c, const float *const K,
                                          const float *const uv_grad_out, const int N, float *xyz_c_grad_in,
                                          cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(xyz_c);
  ASSERT_DEVICE_POINTER(K);
  ASSERT_DEVICE_POINTER(uv_grad_out);
  ASSERT_DEVICE_POINTER(xyz_c_grad_in);

  const int threads_per_block = 256;
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  cam_intr_proj_backward_kernel<<<gridsize, blocksize, 0, stream>>>(xyz_c, K, uv_grad_out, N, xyz_c_grad_in);
}

__global__ void cam_extr_proj_backward_kernel(const float *__restrict__ xyz_w, const float *__restrict__ T,
                                              const float *__restrict__ xyz_c_grad_out, const int N,
                                              float *__restrict__ xyz_w_grad_in) {
  constexpr int XYZ_STRIDE = 3;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = threadIdx.x & 0x1f;

  // Load and broadcast Extrinsic Matrix T (3x4) within warp
  float t_val = 0.0f;
  if (lane_id < 12) {
    t_val = T[lane_id];
  }
  // T = [r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2]
  const float r00 = __shfl_sync(0xffffffff, t_val, 0);
  const float r01 = __shfl_sync(0xffffffff, t_val, 1);
  const float r02 = __shfl_sync(0xffffffff, t_val, 2);
  const float t0 = __shfl_sync(0xffffffff, t_val, 3);
  const float r10 = __shfl_sync(0xffffffff, t_val, 4);
  const float r11 = __shfl_sync(0xffffffff, t_val, 5);
  const float r12 = __shfl_sync(0xffffffff, t_val, 6);
  const float t1 = __shfl_sync(0xffffffff, t_val, 7);
  const float r20 = __shfl_sync(0xffffffff, t_val, 8);
  const float r21 = __shfl_sync(0xffffffff, t_val, 9);
  const float r22 = __shfl_sync(0xffffffff, t_val, 10);
  const float t2 = __shfl_sync(0xffffffff, t_val, 11);

  if (i >= N) {
    return;
  }

  const float grad_x_c = xyz_c_grad_out[i * XYZ_STRIDE + 0];
  const float grad_y_c = xyz_c_grad_out[i * XYZ_STRIDE + 1];
  const float grad_z_c = xyz_c_grad_out[i * XYZ_STRIDE + 2];

  // --- Gradient w.r.t. xyz_w ---
  // d(xyz_w) = R^T * d(xyz_c)
  xyz_w_grad_in[i * XYZ_STRIDE + 0] = r00 * grad_x_c + r10 * grad_y_c + r20 * grad_z_c;
  xyz_w_grad_in[i * XYZ_STRIDE + 1] = r01 * grad_x_c + r11 * grad_y_c + r21 * grad_z_c;
  xyz_w_grad_in[i * XYZ_STRIDE + 2] = r02 * grad_x_c + r12 * grad_y_c + r22 * grad_z_c;
}

void camera_extrinsic_projection_backward(const float *const xyz_w, const float *const T,
                                          const float *const xyz_c_grad_out, const int N, float *xyz_w_grad_in,
                                          cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(xyz_w);
  ASSERT_DEVICE_POINTER(T);
  ASSERT_DEVICE_POINTER(xyz_c_grad_out);
  ASSERT_DEVICE_POINTER(xyz_w_grad_in);

  const int threads_per_block = 256;
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  cam_extr_proj_backward_kernel<<<gridsize, blocksize, 0, stream>>>(xyz_w, T, xyz_c_grad_out, N, xyz_w_grad_in);
}
