// gaussian.cu

#include "checks.cuh"
#include "gsplat/cuda_forward.hpp"

__global__ void form_scale_vector(const float *__restrict__ scale_in, const int N, float *scale_out) {
  constexpr int SCALE_STRIDE = 3;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= N) {
    return;
  }

  float sx = expf(scale_in[SCALE_STRIDE * i + 0]);
  float sy = expf(scale_in[SCALE_STRIDE * i + 2]);
  float sz = expf(scale_in[SCALE_STRIDE * i + 3]);

  scale_out[SCALE_STRIDE * i + 0] = sx * sx;
  scale_out[SCALE_STRIDE * i + 1] = sy * sy;
  scale_out[SCALE_STRIDE * i + 2] = sz * sz;
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
  CHECK_CUDA(cudaMalloc((void **)RS, N * 3 * sizeof(float)));

  float *rotation = nullptr;
  CHECK_CUDA(cudaMalloc((void **)rotation, N * 9 * sizeof(float)));

  float *scale_sqr = nullptr;
  CHECK_CUDA(cudaMalloc((void **)scale_sqr, N * 3 * sizeof(float)));

  // quaternion to rotation matrix
  quat_to_rot_kernel<<<gridsize, blocksize>>>(quaternion, N, rotation);
  // square scale to ensure no negative values
  form_scale_vector<<<gridsize, blocksize>>>(scale, N, scale_sqr);

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // A is rotation (3x3), B is scale_sqr (3x1), C is RS (3x1)
  const int m_rs = 3;
  const int n_rs = 1;
  const int k_rs = 3;

  // RS = rotation x scale_sqr
  CHECK_CUBLAS(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m_rs, n_rs, k_rs, &alpha, rotation, m_rs,
                                         m_rs, scale_sqr, k_rs, k_rs, &beta, RS, m_rs, m_rs, N));

  // A is RS (3x1), B is (RS)^T (1x3), C is sigma (3x3)
  const int m = 3;
  const int n = 3;
  const int k = 1;

  // Sigma = RS x (RS)^T
  CHECK_CUBLAS(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, RS, m, m, RS, k, k, &beta,
                                         sigma, m, m, N));

  CHECK_CUDA(cudaFree(RS));
  CHECK_CUDA(cudaFree(rotation));
  CHECK_CUDA(cudaFree(scale_sqr));
}
