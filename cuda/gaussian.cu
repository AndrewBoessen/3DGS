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

void compute_sigma(float *const quaternion, float *const scale, const int N, float *sigma) {
  ASSERT_DEVICE_POINTER(quaternion);
  ASSERT_DEVICE_POINTER(scale);
  ASSERT_DEVICE_POINTER(sigma);

  const int threads_per_block = 1024;
  // Calculate the number of blocks needed to cover all N points
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  // Launch a single fused kernel to perform all operations.
  // This avoids intermediate memory allocations and global memory traffic.
  compute_sigma_fused_kernel<<<gridsize, blocksize>>>(quaternion, scale, N, sigma);
}
