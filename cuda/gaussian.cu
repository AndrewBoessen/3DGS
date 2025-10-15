// gaussian.cu

#include "checks.cuh"
#include "gsplat/cuda_forward.hpp"

__global__ void compute_sigma_fused_kernel(const float *__restrict__ quaternion, const float *__restrict__ scale,
                                           const int N, float *__restrict__ sigma) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  // Quaternion to Rotation
  const int quat_base_idx = 4 * i;
  float w = quaternion[quat_base_idx + 0];
  float x = quaternion[quat_base_idx + 1];
  float y = quaternion[quat_base_idx + 2];
  float z = quaternion[quat_base_idx + 3];

  const float inv_norm = rsqrtf(w * w + x * x + y * y + z * z);
  w *= inv_norm;
  x *= inv_norm;
  y *= inv_norm;
  z *= inv_norm;

  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  const float xy = x * y;
  const float xz = x * z;
  const float yz = y * z;
  const float wx = w * x;
  const float wy = w * y;
  const float wz = w * z;

  // R (rotation matrix)
  float r00 = 1.0f - 2.0f * (y2 + z2);
  float r01 = 2.0f * (xy - wz);
  float r02 = 2.0f * (xz + wy);
  float r10 = 2.0f * (xy + wz);
  float r11 = 1.0f - 2.0f * (x2 + z2);
  float r12 = 2.0f * (yz - wx);
  float r20 = 2.0f * (xz - wy);
  float r21 = 2.0f * (yz + wx);
  float r22 = 1.0f - 2.0f * (x2 + y2);

  // Apply Scaling
  const int scale_base_idx = 3 * i;
  float sx = expf(scale[scale_base_idx + 0]);
  float sy = expf(scale[scale_base_idx + 1]);
  float sz = expf(scale[scale_base_idx + 2]);

  float rs00 = r00 * sx;
  float rs10 = r10 * sx;
  float rs20 = r20 * sx;

  float rs01 = r01 * sy;
  float rs11 = r11 * sy;
  float rs21 = r21 * sy;

  float rs02 = r02 * sz;
  float rs12 = r12 * sz;
  float rs22 = r22 * sz;

  // Sigma is symmetric, so we can compute the upper-triangular part
  // and reflect it to the lower-triangular part.
  const int sigma_base_idx = 9 * i;
  sigma[sigma_base_idx + 0] = rs00 * rs00 + rs01 * rs01 + rs02 * rs02; // S_00
  sigma[sigma_base_idx + 1] = rs00 * rs10 + rs01 * rs11 + rs02 * rs12; // S_01
  sigma[sigma_base_idx + 2] = rs00 * rs20 + rs01 * rs21 + rs02 * rs22; // S_02
  sigma[sigma_base_idx + 3] = sigma[sigma_base_idx + 1];               // S_10 = S_01
  sigma[sigma_base_idx + 4] = rs10 * rs10 + rs11 * rs11 + rs12 * rs12; // S_11
  sigma[sigma_base_idx + 5] = rs10 * rs20 + rs11 * rs21 + rs12 * rs22; // S_12
  sigma[sigma_base_idx + 6] = sigma[sigma_base_idx + 2];               // S_20 = S_02
  sigma[sigma_base_idx + 7] = sigma[sigma_base_idx + 5];               // S_21 = S_12
  sigma[sigma_base_idx + 8] = rs20 * rs20 + rs21 * rs21 + rs22 * rs22; // S_22
}

__global__ void compute_conic_kernel(const float *__restrict__ sigma, const float *__restrict__ T,
                                     const float *__restrict__ J, const int N, float *conic) {
  constexpr int SIGMA_STRIDE = 9;
  constexpr int J_STRIDE = 6;
  constexpr int CONIC_STRIDE = 3;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = threadIdx.x & 0x1f; // lane_id in warp (0-31)

  // Load and broadcast Extrinsic Matrix T (3x4) within warp
  float t_val = 0.0f;
  if (lane_id < 12) {
    t_val = T[lane_id];
  }
  // T = [r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2]
  // W = [r00, r01, r02, r10, r11, r12, r20, r21, r22]
  const float w00 = __shfl_sync(0xffffffff, t_val, 0);
  const float w01 = __shfl_sync(0xffffffff, t_val, 1);
  const float w02 = __shfl_sync(0xffffffff, t_val, 2);
  const float w10 = __shfl_sync(0xffffffff, t_val, 4);
  const float w11 = __shfl_sync(0xffffffff, t_val, 5);
  const float w12 = __shfl_sync(0xffffffff, t_val, 6);
  const float w20 = __shfl_sync(0xffffffff, t_val, 8);
  const float w21 = __shfl_sync(0xffffffff, t_val, 9);
  const float w22 = __shfl_sync(0xffffffff, t_val, 10);

  if (i >= N) {
    return;
  }

  // Compute conic = JW @ Sigma @ (JW)^T

  // Load the per-Gaussian 3x3 symmetric covariance matrix (Sigma) into registers.
  // Only load the 6 unique elements to save bandwidth.
  const int sigma_base_idx = i * SIGMA_STRIDE;
  const float s00 = sigma[sigma_base_idx + 0];
  const float s01 = sigma[sigma_base_idx + 1];
  const float s02 = sigma[sigma_base_idx + 2];
  const float s11 = sigma[sigma_base_idx + 4];
  const float s12 = sigma[sigma_base_idx + 5];
  const float s22 = sigma[sigma_base_idx + 8];

  // Load the per-Gaussian 2x3 projection Jacobian (J) into registers.
  const int j_base_idx = i * J_STRIDE;
  const float j00 = J[j_base_idx + 0];
  const float j01 = J[j_base_idx + 1];
  const float j02 = J[j_base_idx + 2];
  const float j10 = J[j_base_idx + 3];
  const float j11 = J[j_base_idx + 4];
  const float j12 = J[j_base_idx + 5];

  // 1. Compute M = J @ W. M is a 2x3 matrix.
  const float m00 = j00 * w00 + j01 * w10 + j02 * w20;
  const float m01 = j00 * w01 + j01 * w11 + j02 * w21;
  const float m02 = j00 * w02 + j01 * w12 + j02 * w22;
  const float m10 = j10 * w00 + j11 * w10 + j12 * w20;
  const float m11 = j10 * w01 + j11 * w11 + j12 * w21;
  const float m12 = j10 * w02 + j11 * w12 + j12 * w22;

  // 2. Compute V = Sigma @ M^T. V is a 3x2 matrix.
  const float v00 = s00 * m00 + s01 * m01 + s02 * m02;
  const float v01 = s00 * m10 + s01 * m11 + s02 * m12;
  const float v10 = s01 * m00 + s11 * m01 + s12 * m02;
  const float v11 = s01 * m10 + s11 * m11 + s12 * m12;
  const float v20 = s02 * m00 + s12 * m01 + s22 * m02;
  const float v21 = s02 * m10 + s12 * m11 + s22 * m12;

  // 3. Compute conic = M @ V. The resulting conic is a 2x2 symmetric matrix.
  // We only need to compute and store the 3 unique elements of the upper triangle.
  const float c00 = m00 * v00 + m01 * v10 + m02 * v20;
  const float c01 = m00 * v01 + m01 * v11 + m02 * v21; // Also equals c10
  const float c11 = m10 * v01 + m11 * v11 + m12 * v21;

  // 4. Store the 3 unique components of the conic matrix into global memory.
  const int conic_base_idx = i * CONIC_STRIDE;
  conic[conic_base_idx + 0] = c00;
  conic[conic_base_idx + 1] = c01;
  conic[conic_base_idx + 2] = c11;
}

__global__ void compute_projection_jacobian_kernel(const float *__restrict__ xyz, const float *__restrict__ K,
                                                   const int N, float *J) {
  constexpr int XYZ_STRIDE = 3;
  constexpr int J_STRIDE = 6;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = threadIdx.x & 0x1f;

  // load and broadcast K to all threads in warp
  float k_val = 0.0f;
  if (lane_id < 9) {
    k_val = K[lane_id];
  }
  // K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
  const float fx = __shfl_sync(0xffffffff, k_val, 0);
  const float fy = __shfl_sync(0xffffffff, k_val, 4);

  if (i >= N) {
    return;
  }

  float x = xyz[i * XYZ_STRIDE + 0];
  float y = xyz[i * XYZ_STRIDE + 1];
  float z = xyz[i * XYZ_STRIDE + 2];

  J[i * J_STRIDE + 0] = fx / z;
  J[i * J_STRIDE + 1] = 0;
  J[i * J_STRIDE + 2] = -fx * x / (z * z);
  J[i * J_STRIDE + 3] = 0;
  J[i * J_STRIDE + 4] = fy / z;
  J[i * J_STRIDE + 5] = -fy * y / (z * z);
}

void compute_sigma(float *const quaternion, float *const scale, const int N, float *sigma, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(quaternion);
  ASSERT_DEVICE_POINTER(scale);
  ASSERT_DEVICE_POINTER(sigma);

  const int threads_per_block = 256;
  // Calculate the number of blocks needed to cover all N points
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  // Launch a single fused kernel to perform all operations.
  // This avoids intermediate memory allocations and global memory traffic.
  compute_sigma_fused_kernel<<<gridsize, blocksize, 0, stream>>>(quaternion, scale, N, sigma);
}

void compute_conic(float *const xyz, const float *K, float *const sigma, const float *T, const int N, float *J,
                   float *conic, cudaStream_t stream) {
  // Ensure all provided pointers are valid GPU device pointers.
  ASSERT_DEVICE_POINTER(xyz);
  ASSERT_DEVICE_POINTER(K);
  ASSERT_DEVICE_POINTER(sigma);
  ASSERT_DEVICE_POINTER(T);
  ASSERT_DEVICE_POINTER(J);
  ASSERT_DEVICE_POINTER(conic);

  const int threads_per_block = 256;

  // Calculate the number of blocks needed to cover all N Gaussians.
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  const dim3 gridsize(num_blocks, 1, 1);
  const dim3 blocksize(threads_per_block, 1, 1);

  // This kernel computes the Jacobian (J) for each Gaussian.
  compute_projection_jacobian_kernel<<<gridsize, blocksize, 0, stream>>>(xyz, K, N, J);

  // This kernel uses the world-space covariance (sigma), the camera transform (T),
  // and the Jacobian (J) computed in the previous step to find the 2D conic.
  compute_conic_kernel<<<gridsize, blocksize, 0, stream>>>(sigma, T, J, N, conic);
}
