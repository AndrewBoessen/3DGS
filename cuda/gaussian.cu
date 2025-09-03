// gaussian.cu

#include "checks.cuh"
#include "gsplat/cuda_forward.hpp"

__global__ void scale_rotation_kernel(const float *__restrict__ rotation, const float *__restrict__ scale, const int N,
                                      float *RS) {
  constexpr int SCALE_STRIDE = 3;
  constexpr int ROTATION_STRIDE = 9;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= N) {
    return;
  }

  float sx = expf(scale[SCALE_STRIDE * i + 0]);
  float sy = expf(scale[SCALE_STRIDE * i + 1]);
  float sz = expf(scale[SCALE_STRIDE * i + 2]);

  const int base_idx = ROTATION_STRIDE * i;

  // --- Column 1 of RS ---
  RS[base_idx + 0] = rotation[base_idx + 0] * sx; // R(0,0)
  RS[base_idx + 1] = rotation[base_idx + 3] * sy; // R(1,0)
  RS[base_idx + 2] = rotation[base_idx + 6] * sz; // R(2,0)

  // --- Column 2 of RS ---
  RS[base_idx + 3] = rotation[base_idx + 1] * sx; // R(0,1)
  RS[base_idx + 4] = rotation[base_idx + 4] * sy; // R(1,1)
  RS[base_idx + 5] = rotation[base_idx + 7] * sz; // R(2,1)

  // --- Column 3 of RS ---
  RS[base_idx + 6] = rotation[base_idx + 2] * sx; // R(0,2)
  RS[base_idx + 7] = rotation[base_idx + 5] * sy; // R(1,2)
  RS[base_idx + 8] = rotation[base_idx + 8] * sz; // R(2,2)
}

__global__ void quat_to_rot_kernel(const float *__restrict__ quaternion, const int N, float *rotation) {
  constexpr int QUAT_STRIDE = 4;
  constexpr int ROT_STRIDE = 9;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= N) {
    return;
  }

  float w = quaternion[QUAT_STRIDE * i + 0];
  float x = quaternion[QUAT_STRIDE * i + 1];
  float y = quaternion[QUAT_STRIDE * i + 2];
  float z = quaternion[QUAT_STRIDE * i + 3];

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

  // Formula for converting a quaternion (w, x, y, z) to a rotation matrix:
  // | 1 - 2(y^2 + z^2)   2(xy - wz)       2(xz + wy)     |
  // | 2(xy + wz)         1 - 2(x^2 + z^2)   2(yz - wx)     |
  // | 2(xz - wy)         2(yz + wx)       1 - 2(x^2 + y^2) |

  // Row 0
  rotation[ROT_STRIDE * i + 0] = 1.0f - 2.0f * (y2 + z2);
  rotation[ROT_STRIDE * i + 1] = 2.0f * (xy - wz);
  rotation[ROT_STRIDE * i + 2] = 2.0f * (xz + wy);

  // Row 1
  rotation[ROT_STRIDE * i + 3] = 2.0f * (xy + wz);
  rotation[ROT_STRIDE * i + 4] = 1.0f - 2.0f * (x2 + z2);
  rotation[ROT_STRIDE * i + 5] = 2.0f * (yz - wx);

  // Row 2
  rotation[ROT_STRIDE * i + 6] = 2.0f * (xz - wy);
  rotation[ROT_STRIDE * i + 7] = 2.0f * (yz + wx);
  rotation[ROT_STRIDE * i + 8] = 1.0f - 2.0f * (x2 + y2);
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

  float *RS = nullptr;
  CHECK_CUDA(cudaMalloc(&RS, N * 9 * sizeof(float)));

  float *rotation = nullptr;
  CHECK_CUDA(cudaMalloc(&rotation, N * 9 * sizeof(float)));

  // quaternion to rotation matrix
  quat_to_rot_kernel<<<gridsize, blocksize>>>(quaternion, N, rotation);
  // scale rotation matrix
  scale_rotation_kernel<<<gridsize, blocksize>>>(rotation, scale, N, RS);

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // A is RS (3x3), B is (RS)^T (3x3), C is sigma (3x3)
  const int m = 3;
  const int n = 3;
  const int k = 3;

  // Sigma = RS x (RS)^T
  CHECK_CUBLAS(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, RS, m, m * k, RS, n, n * k,
                                         &beta, sigma, m, m * n, N));

  CHECK_CUDA(cudaFree(RS));
  CHECK_CUDA(cudaFree(rotation));
}
