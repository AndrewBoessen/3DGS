// projection_backward.cu

#include "checks.cuh"
#include "gsplat/cuda_backward.hpp"

__global__ void compute_proj_jacobian_backward_kernel(const float *__restrict__ xyz_c, const float *__restrict__ K,
                                                      const float *__restrict__ J_grad_in, const int N,
                                                      float *__restrict__ xyz_c_grad_out,
                                                      float *__restrict__ K_grad_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  const int lane_id = threadIdx.x & 0x1f;
  float k_val = 0.0f;
  if (lane_id < 4)
    k_val = K[lane_id];
  const float fx = __shfl_sync(0xffffffff, k_val, 0);
  const float fy = __shfl_sync(0xffffffff, k_val, 2);

  const float x = xyz_c[i * 3 + 0];
  const float y = xyz_c[i * 3 + 1];
  const float z = xyz_c[i * 3 + 2];

  if (z <= 1e-6f) {
    xyz_c_grad_out[i * 3 + 0] = 0.0f;
    xyz_c_grad_out[i * 3 + 1] = 0.0f;
    xyz_c_grad_out[i * 3 + 2] = 0.0f;
    return;
  }

  const float z_inv = 1.0f / z;
  const float z_inv2 = z_inv * z_inv;
  const float z_inv3 = z_inv2 * z_inv;

  const float *grad_J = J_grad_in + i * 6;

  // Gradient w.r.t. K (fx, fy)
  atomicAdd(&K_grad_out[0], grad_J[0] * z_inv - grad_J[2] * x * z_inv2); // d_fx
  atomicAdd(&K_grad_out[2], grad_J[4] * z_inv - grad_J[5] * y * z_inv2); // d_fy

  // Gradient w.r.t. xyz_c
  float gx = -grad_J[2] * fx * z_inv2;
  float gy = -grad_J[5] * fy * z_inv2;
  float gz = -grad_J[0] * fx * z_inv2 + grad_J[2] * 2.0f * fx * x * z_inv3 - grad_J[4] * fy * z_inv2 +
             grad_J[5] * 2.0f * fy * y * z_inv3;

  atomicAdd(&xyz_c_grad_out[i * 3 + 0], gx);
  atomicAdd(&xyz_c_grad_out[i * 3 + 1], gy);
  atomicAdd(&xyz_c_grad_out[i * 3 + 2], gz);
}

void compute_projection_jacobian_backward(const float *const xyz_c, const float *const K, const float *const J_grad_in,
                                          const int N, float *xyz_c_grad_out, float *K_grad_out, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(xyz_c);
  ASSERT_DEVICE_POINTER(K);
  ASSERT_DEVICE_POINTER(J_grad_in);
  ASSERT_DEVICE_POINTER(xyz_c_grad_out);
  ASSERT_DEVICE_POINTER(K_grad_out);

  // K_grad_out is an accumulation over all points, so it must be zeroed.
  CHECK_CUDA(cudaMemsetAsync(K_grad_out, 0, 4 * sizeof(float), stream));

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  compute_proj_jacobian_backward_kernel<<<blocks, threads, 0, stream>>>(xyz_c, K, J_grad_in, N, xyz_c_grad_out,
                                                                        K_grad_out);
}

__global__ void compute_conic_backward_kernel(const float *__restrict__ J_in, const float *__restrict__ sigma_in,
                                              const float *__restrict__ T, const float *__restrict__ conic_grad_out,
                                              const int N, float *__restrict__ J_grad_in,
                                              float *__restrict__ sigma_grad_in) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  // Unpack inputs
  const float J[] = {J_in[i * 6], J_in[i * 6 + 1], J_in[i * 6 + 2], J_in[i * 6 + 3], J_in[i * 6 + 4], J_in[i * 6 + 5]};
  const float sigma[] = {sigma_in[i * 6],     sigma_in[i * 6 + 1], sigma_in[i * 6 + 2],
                         sigma_in[i * 6 + 3], sigma_in[i * 6 + 4], sigma_in[i * 6 + 5]};
  const float grad_conic[] = {conic_grad_out[i * 3], conic_grad_out[i * 3 + 1], conic_grad_out[i * 3 + 2]};

  const float R[] = {T[0], T[1], T[2], T[4], T[5], T[6], T[8], T[9], T[10]};

  // Sigma_cam = R * Sigma * R.T
  // Intermediate calculations for Sigma_cam = R * Sigma
  float RS[9];
  RS[0] = R[0] * sigma[0] + R[1] * sigma[1] + R[2] * sigma[2];
  RS[1] = R[0] * sigma[1] + R[1] * sigma[3] + R[2] * sigma[4];
  RS[2] = R[0] * sigma[2] + R[1] * sigma[4] + R[2] * sigma[5];
  RS[3] = R[3] * sigma[0] + R[4] * sigma[1] + R[5] * sigma[2];
  RS[4] = R[3] * sigma[1] + R[4] * sigma[3] + R[5] * sigma[4];
  RS[5] = R[3] * sigma[2] + R[4] * sigma[4] + R[5] * sigma[5];
  RS[6] = R[6] * sigma[0] + R[7] * sigma[1] + R[8] * sigma[2];
  RS[7] = R[6] * sigma[1] + R[7] * sigma[3] + R[8] * sigma[4];
  RS[8] = R[6] * sigma[2] + R[7] * sigma[4] + R[8] * sigma[5];

  float Sigma_cam[9];
  Sigma_cam[0] = RS[0] * R[0] + RS[1] * R[1] + RS[2] * R[2];
  Sigma_cam[1] = RS[0] * R[3] + RS[1] * R[4] + RS[2] * R[5];
  Sigma_cam[2] = RS[0] * R[6] + RS[1] * R[7] + RS[2] * R[8];
  Sigma_cam[3] = RS[3] * R[0] + RS[4] * R[1] + RS[5] * R[2];
  Sigma_cam[4] = RS[3] * R[3] + RS[4] * R[4] + RS[5] * R[5];
  Sigma_cam[5] = RS[3] * R[6] + RS[4] * R[7] + RS[5] * R[8];
  Sigma_cam[6] = RS[6] * R[0] + RS[7] * R[1] + RS[8] * R[2];
  Sigma_cam[7] = RS[6] * R[3] + RS[7] * R[4] + RS[8] * R[5];
  Sigma_cam[8] = RS[6] * R[6] + RS[7] * R[7] + RS[8] * R[8];

  // grad_conic_mat is symmetric
  const float grad_conic_mat[] = {grad_conic[0], grad_conic[1], grad_conic[1], grad_conic[2]};

  // grad_Sigma_cam = J.T * grad_conic_mat * J
  float grad_Sigma_cam[9];
  float J_T_grad_C[6];
  J_T_grad_C[0] = J[0] * grad_conic_mat[0] + J[3] * grad_conic_mat[2];
  J_T_grad_C[1] = J[0] * grad_conic_mat[1] + J[3] * grad_conic_mat[3];
  J_T_grad_C[2] = J[1] * grad_conic_mat[0] + J[4] * grad_conic_mat[2];
  J_T_grad_C[3] = J[1] * grad_conic_mat[1] + J[4] * grad_conic_mat[3];
  J_T_grad_C[4] = J[2] * grad_conic_mat[0] + J[5] * grad_conic_mat[2];
  J_T_grad_C[5] = J[2] * grad_conic_mat[1] + J[5] * grad_conic_mat[3];

  grad_Sigma_cam[0] = J_T_grad_C[0] * J[0] + J_T_grad_C[1] * J[3];
  grad_Sigma_cam[1] = J_T_grad_C[0] * J[1] + J_T_grad_C[1] * J[4];
  grad_Sigma_cam[2] = J_T_grad_C[0] * J[2] + J_T_grad_C[1] * J[5];
  grad_Sigma_cam[3] = J_T_grad_C[2] * J[0] + J_T_grad_C[3] * J[3];
  grad_Sigma_cam[4] = J_T_grad_C[2] * J[1] + J_T_grad_C[3] * J[4];
  grad_Sigma_cam[5] = J_T_grad_C[2] * J[2] + J_T_grad_C[3] * J[5];
  grad_Sigma_cam[6] = J_T_grad_C[4] * J[0] + J_T_grad_C[5] * J[3];
  grad_Sigma_cam[7] = J_T_grad_C[4] * J[1] + J_T_grad_C[5] * J[4];
  grad_Sigma_cam[8] = J_T_grad_C[4] * J[2] + J_T_grad_C[5] * J[5];

  // grad_sigma = R.T * grad_Sigma_cam * R
  float R_T_grad_Scam[9];
  R_T_grad_Scam[0] = R[0] * grad_Sigma_cam[0] + R[3] * grad_Sigma_cam[3] + R[6] * grad_Sigma_cam[6];
  R_T_grad_Scam[1] = R[0] * grad_Sigma_cam[1] + R[3] * grad_Sigma_cam[4] + R[6] * grad_Sigma_cam[7];
  R_T_grad_Scam[2] = R[0] * grad_Sigma_cam[2] + R[3] * grad_Sigma_cam[5] + R[6] * grad_Sigma_cam[8];
  R_T_grad_Scam[3] = R[1] * grad_Sigma_cam[0] + R[4] * grad_Sigma_cam[3] + R[7] * grad_Sigma_cam[6];
  R_T_grad_Scam[4] = R[1] * grad_Sigma_cam[1] + R[4] * grad_Sigma_cam[4] + R[7] * grad_Sigma_cam[7];
  R_T_grad_Scam[5] = R[1] * grad_Sigma_cam[2] + R[4] * grad_Sigma_cam[5] + R[7] * grad_Sigma_cam[8];
  R_T_grad_Scam[6] = R[2] * grad_Sigma_cam[0] + R[5] * grad_Sigma_cam[3] + R[8] * grad_Sigma_cam[6];
  R_T_grad_Scam[7] = R[2] * grad_Sigma_cam[1] + R[5] * grad_Sigma_cam[4] + R[8] * grad_Sigma_cam[7];
  R_T_grad_Scam[8] = R[2] * grad_Sigma_cam[2] + R[5] * grad_Sigma_cam[5] + R[8] * grad_Sigma_cam[8];

  sigma_grad_in[i * 6 + 0] = R_T_grad_Scam[0] * R[0] + R_T_grad_Scam[1] * R[3] + R_T_grad_Scam[2] * R[6];
  sigma_grad_in[i * 6 + 1] = R_T_grad_Scam[0] * R[1] + R_T_grad_Scam[1] * R[4] + R_T_grad_Scam[2] * R[7];
  sigma_grad_in[i * 6 + 2] = R_T_grad_Scam[0] * R[2] + R_T_grad_Scam[1] * R[5] + R_T_grad_Scam[2] * R[8];
  sigma_grad_in[i * 6 + 3] = R_T_grad_Scam[3] * R[1] + R_T_grad_Scam[4] * R[4] + R_T_grad_Scam[5] * R[7];
  sigma_grad_in[i * 6 + 4] = R_T_grad_Scam[3] * R[2] + R_T_grad_Scam[4] * R[5] + R_T_grad_Scam[5] * R[8];
  sigma_grad_in[i * 6 + 5] = R_T_grad_Scam[6] * R[2] + R_T_grad_Scam[7] * R[5] + R_T_grad_Scam[8] * R[8];

  // grad_J = 2 * grad_conic_mat * J * Sigma_cam
  float grad_J_mat[6];
  float J_Scam[6];
  J_Scam[0] = J[0] * Sigma_cam[0] + J[1] * Sigma_cam[3] + J[2] * Sigma_cam[6];
  J_Scam[1] = J[0] * Sigma_cam[1] + J[1] * Sigma_cam[4] + J[2] * Sigma_cam[7];
  J_Scam[2] = J[0] * Sigma_cam[2] + J[1] * Sigma_cam[5] + J[2] * Sigma_cam[8];
  J_Scam[3] = J[3] * Sigma_cam[0] + J[4] * Sigma_cam[3] + J[5] * Sigma_cam[6];
  J_Scam[4] = J[3] * Sigma_cam[1] + J[4] * Sigma_cam[4] + J[5] * Sigma_cam[7];
  J_Scam[5] = J[3] * Sigma_cam[2] + J[4] * Sigma_cam[5] + J[5] * Sigma_cam[8];

  grad_J_mat[0] = grad_conic_mat[0] * J_Scam[0] + grad_conic_mat[1] * J_Scam[3];
  grad_J_mat[1] = grad_conic_mat[0] * J_Scam[1] + grad_conic_mat[1] * J_Scam[4];
  grad_J_mat[2] = grad_conic_mat[0] * J_Scam[2] + grad_conic_mat[1] * J_Scam[5];
  grad_J_mat[3] = grad_conic_mat[2] * J_Scam[0] + grad_conic_mat[3] * J_Scam[3];
  grad_J_mat[4] = grad_conic_mat[2] * J_Scam[1] + grad_conic_mat[3] * J_Scam[4];
  grad_J_mat[5] = grad_conic_mat[2] * J_Scam[2] + grad_conic_mat[3] * J_Scam[5];

  for (int j = 0; j < 6; ++j)
    J_grad_in[i * 6 + j] = 2.0f * grad_J_mat[j];
}

void compute_conic_backward(const float *const J, const float *const sigma, const float *const T,
                            const float *const conic_grad_out, const int N, float *J_grad_in, float *sigma_grad_in,
                            cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(J);
  ASSERT_DEVICE_POINTER(sigma);
  ASSERT_DEVICE_POINTER(T);
  ASSERT_DEVICE_POINTER(conic_grad_out);
  ASSERT_DEVICE_POINTER(J_grad_in);
  ASSERT_DEVICE_POINTER(sigma_grad_in);

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  compute_conic_backward_kernel<<<blocks, threads, 0, stream>>>(J, sigma, T, conic_grad_out, N, J_grad_in,
                                                                sigma_grad_in);
}

__global__ void compute_sigma_backward_kernel(const float *__restrict__ quaternion, const float *__restrict__ scale,
                                              const float *__restrict__ sigma_grad_in, const int N,
                                              float *__restrict__ quaternion_grad_out,
                                              float *__restrict__ scale_grad_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  // --- Recompute forward pass variables ---
  float w = quaternion[i * 4 + 0], x = quaternion[i * 4 + 1], y = quaternion[i * 4 + 2], z = quaternion[i * 4 + 3];
  const float inv_norm = rsqrtf(w * w + x * x + y * y + z * z);
  w *= inv_norm;
  x *= inv_norm;
  y *= inv_norm;
  z *= inv_norm;

  // R (rotation matrix)
  const float R[] = {1.f - 2.f * (y * y + z * z), 2.f * (x * y - w * z),       2.f * (x * z + w * y),
                     2.f * (x * y + w * z),       1.f - 2.f * (x * x + z * z), 2.f * (y * z - w * x),
                     2.f * (x * z - w * y),       2.f * (y * z + w * x),       1.f - 2.f * (x * x + y * y)};

  // S (scale matrix diagonal)
  const float sx = scale[i * 3 + 0], sy = scale[i * 3 + 1], sz = scale[i * 3 + 2];

  // --- Backward Pass ---
  const float *grad_Sigma = sigma_grad_in + i * 6;
  const float grad_Sigma_mat[] = {// Unpack symmetric matrix
                                  grad_Sigma[0], grad_Sigma[1], grad_Sigma[2], grad_Sigma[1], grad_Sigma[3],
                                  grad_Sigma[4], grad_Sigma[2], grad_Sigma[4], grad_Sigma[5]};

  // grad_M = R.T * grad_Sigma * R (M = S*S)
  float R_T_grad_S[9];
  R_T_grad_S[0] = R[0] * grad_Sigma_mat[0] + R[3] * grad_Sigma_mat[3] + R[6] * grad_Sigma_mat[6];
  R_T_grad_S[1] = R[0] * grad_Sigma_mat[1] + R[3] * grad_Sigma_mat[4] + R[6] * grad_Sigma_mat[7];
  R_T_grad_S[2] = R[0] * grad_Sigma_mat[2] + R[3] * grad_Sigma_mat[5] + R[6] * grad_Sigma_mat[8];
  R_T_grad_S[3] = R[1] * grad_Sigma_mat[0] + R[4] * grad_Sigma_mat[3] + R[7] * grad_Sigma_mat[6];
  R_T_grad_S[4] = R[1] * grad_Sigma_mat[1] + R[4] * grad_Sigma_mat[4] + R[7] * grad_Sigma_mat[7];
  R_T_grad_S[5] = R[1] * grad_Sigma_mat[2] + R[4] * grad_Sigma_mat[5] + R[7] * grad_Sigma_mat[8];
  R_T_grad_S[6] = R[2] * grad_Sigma_mat[0] + R[5] * grad_Sigma_mat[3] + R[8] * grad_Sigma_mat[6];
  R_T_grad_S[7] = R[2] * grad_Sigma_mat[1] + R[5] * grad_Sigma_mat[4] + R[8] * grad_Sigma_mat[7];
  R_T_grad_S[8] = R[2] * grad_Sigma_mat[2] + R[5] * grad_Sigma_mat[5] + R[8] * grad_Sigma_mat[8];

  float grad_M[9];
  grad_M[0] = R_T_grad_S[0] * R[0] + R_T_grad_S[1] * R[3] + R_T_grad_S[2] * R[6];
  grad_M[4] = R_T_grad_S[3] * R[1] + R_T_grad_S[4] * R[4] + R_T_grad_S[5] * R[7];
  grad_M[8] = R_T_grad_S[6] * R[2] + R_T_grad_S[7] * R[5] + R_T_grad_S[8] * R[8];

  // grad_scale
  scale_grad_out[i * 3 + 0] = 2.0f * sx * grad_M[0];
  scale_grad_out[i * 3 + 1] = 2.0f * sy * grad_M[4];
  scale_grad_out[i * 3 + 2] = 2.0f * sz * grad_M[8];

  // grad_R = (grad_Sigma + grad_Sigma.T) * R * M
  const float M[] = {sx * sx, sy * sy, sz * sz};
  const float grad_Sigma_sym[] = {2 * grad_Sigma[0], 2 * grad_Sigma[1], 2 * grad_Sigma[2],
                                  2 * grad_Sigma[3], 2 * grad_Sigma[4], 2 * grad_Sigma[5]};
  float gS_R[9];
  gS_R[0] = grad_Sigma_sym[0] * R[0] + grad_Sigma_sym[1] * R[3] + grad_Sigma_sym[2] * R[6];
  gS_R[1] = grad_Sigma_sym[0] * R[1] + grad_Sigma_sym[1] * R[4] + grad_Sigma_sym[2] * R[7];
  gS_R[2] = grad_Sigma_sym[0] * R[2] + grad_Sigma_sym[1] * R[5] + grad_Sigma_sym[2] * R[8];
  gS_R[3] = grad_Sigma_sym[1] * R[0] + grad_Sigma_sym[3] * R[3] + grad_Sigma_sym[4] * R[6];
  gS_R[4] = grad_Sigma_sym[1] * R[1] + grad_Sigma_sym[3] * R[4] + grad_Sigma_sym[4] * R[7];
  gS_R[5] = grad_Sigma_sym[1] * R[2] + grad_Sigma_sym[3] * R[5] + grad_Sigma_sym[4] * R[8];
  gS_R[6] = grad_Sigma_sym[2] * R[0] + grad_Sigma_sym[4] * R[3] + grad_Sigma_sym[5] * R[6];
  gS_R[7] = grad_Sigma_sym[2] * R[1] + grad_Sigma_sym[4] * R[4] + grad_Sigma_sym[5] * R[7];
  gS_R[8] = grad_Sigma_sym[2] * R[2] + grad_Sigma_sym[4] * R[5] + grad_Sigma_sym[5] * R[8];

  float grad_R[9];
  grad_R[0] = gS_R[0] * M[0];
  grad_R[1] = gS_R[1] * M[1];
  grad_R[2] = gS_R[2] * M[2];
  grad_R[3] = gS_R[3] * M[0];
  grad_R[4] = gS_R[4] * M[1];
  grad_R[5] = gS_R[5] * M[2];
  grad_R[6] = gS_R[6] * M[0];
  grad_R[7] = gS_R[7] * M[1];
  grad_R[8] = gS_R[8] * M[2];

  // grad_quaternion
  // These are derived from the chain rule dL/dq = dL/dR : dR/dq
  float gw, gx, gy, gz;
  gw = 2.f * (-grad_R[3] * z + grad_R[6] * y - grad_R[1] * z + grad_R[7] * x + grad_R[2] * y - grad_R[5] * x);
  gx = 2.f * (grad_R[4] * y + grad_R[7] * z + grad_R[5] * z + grad_R[8] * y - grad_R[1] * y - grad_R[2] * z);
  gy = 2.f * (grad_R[0] * y - grad_R[3] * x + grad_R[8] * z - grad_R[6] * x + grad_R[2] * w - grad_R[5] * w);
  gz = 2.f * (-grad_R[0] * z + grad_R[6] * w - grad_R[3] * w + grad_R[1] * x + grad_R[4] * z - grad_R[7] * x);

  quaternion_grad_out[i * 4 + 0] = gw;
  quaternion_grad_out[i * 4 + 1] = gx;
  quaternion_grad_out[i * 4 + 2] = gy;
  quaternion_grad_out[i * 4 + 3] = gz;
}

void compute_sigma_backward(const float *const quaternion, const float *const scale, const float *const sigma_grad_in,
                            const int N, float *quaternion_grad_out, float *scale_grad_out, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(quaternion);
  ASSERT_DEVICE_POINTER(scale);
  ASSERT_DEVICE_POINTER(sigma_grad_in);
  ASSERT_DEVICE_POINTER(quaternion_grad_out);
  ASSERT_DEVICE_POINTER(scale_grad_out);

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  compute_sigma_backward_kernel<<<blocks, threads, 0, stream>>>(quaternion, scale, sigma_grad_in, N,
                                                                quaternion_grad_out, scale_grad_out);
}
