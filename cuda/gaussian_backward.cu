// projection_backward.cu

#include "checks.cuh"
#include "gsplat/cuda_backward.hpp"

__global__ void compute_proj_jacobian_backward_kernel(const float *__restrict__ xyz_c, const float *__restrict__ K,
                                                      const float *__restrict__ J_grad_out, const int N,
                                                      float *__restrict__ xyz_c_grad_in) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  const int lane_id = threadIdx.x & 0x1f;
  float k_val = 0.0f;
  if (lane_id < 9)
    k_val = K[lane_id];
  const float fx = __shfl_sync(0xffffffff, k_val, 0);
  const float fy = __shfl_sync(0xffffffff, k_val, 4);

  if (i >= N)
    return;

  const float x = xyz_c[i * 3 + 0];
  const float y = xyz_c[i * 3 + 1];
  const float z = xyz_c[i * 3 + 2];

  if (z <= 1e-6f) {
    xyz_c_grad_in[i * 3 + 0] = 0.0f;
    xyz_c_grad_in[i * 3 + 1] = 0.0f;
    xyz_c_grad_in[i * 3 + 2] = 0.0f;
    return;
  }

  const float z_inv = 1.0f / z;
  const float z_inv2 = z_inv * z_inv;
  const float z_inv3 = z_inv2 * z_inv;

  const float *grad_J = J_grad_out + i * 6;

  // Gradient w.r.t. xyz_c
  float gx = -grad_J[2] * fx * z_inv2;
  float gy = -grad_J[5] * fy * z_inv2;
  float gz = -grad_J[0] * fx * z_inv2 + grad_J[2] * 2.0f * fx * x * z_inv3 - grad_J[4] * fy * z_inv2 +
             grad_J[5] * 2.0f * fy * y * z_inv3;

  xyz_c_grad_in[i * 3 + 0] = gx;
  xyz_c_grad_in[i * 3 + 1] = gy;
  xyz_c_grad_in[i * 3 + 2] = gz;
}

void compute_projection_jacobian_backward(const float *const xyz_c, const float *const K, const float *const J_grad_out,
                                          const int N, float *xyz_c_grad_in, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(xyz_c);
  ASSERT_DEVICE_POINTER(K);
  ASSERT_DEVICE_POINTER(J_grad_out);
  ASSERT_DEVICE_POINTER(xyz_c_grad_in);

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  compute_proj_jacobian_backward_kernel<<<blocks, threads, 0, stream>>>(xyz_c, K, J_grad_out, N, xyz_c_grad_in);
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

__global__ void sigma_backward_kernel(const float *__restrict__ q, const float *__restrict__ s,
                                      const float *__restrict__ dSigma_in, int N, float *__restrict__ dQ_in,
                                      float *__restrict__ dS_in) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  // Load quaternion and scale for the current Gaussian
  const float qw = q[idx * 4 + 0];
  const float qx = q[idx * 4 + 1];
  const float qy = q[idx * 4 + 2];
  const float qz = q[idx * 4 + 3];

  const float sx = s[idx * 3 + 0];
  const float sy = s[idx * 3 + 1];
  const float sz = s[idx * 3 + 2];

  // --- 1. Recompute intermediate variables from the forward pass ---

  // Normalize quaternion
  const float norm = sqrtf(qw * qw + qx * qx + qy * qy + qz * qz) + 1e-8f;
  const float w = qw / norm;
  const float x = qx / norm;
  const float y = qy / norm;
  const float z = qz / norm;

  // Exponentiate scales
  const float S_x = expf(sx);
  const float S_y = expf(sy);
  const float S_z = expf(sz);

  // Rotation matrix R from normalized quaternion
  float R[9];
  R[0] = 1.0f - 2.0f * (y * y + z * z);
  R[1] = 2.0f * (x * y - w * z);
  R[2] = 2.0f * (x * z + w * y);
  R[3] = 2.0f * (x * y + w * z);
  R[4] = 1.0f - 2.0f * (x * x + z * z);
  R[5] = 2.0f * (y * z - w * x);
  R[6] = 2.0f * (x * z - w * y);
  R[7] = 2.0f * (y * z + w * x);
  R[8] = 1.0f - 2.0f * (x * x + y * y);

  // M = R * S
  float M[9];
  M[0] = R[0] * S_x;
  M[1] = R[1] * S_y;
  M[2] = R[2] * S_z;
  M[3] = R[3] * S_x;
  M[4] = R[4] * S_y;
  M[5] = R[5] * S_z;
  M[6] = R[6] * S_x;
  M[7] = R[7] * S_y;
  M[8] = R[8] * S_z;

  // --- 2. Backpropagate ---

  // Load dSigma and reconstruct the full symmetric matrix
  float dSigma[9];
  dSigma[0] = dSigma_in[idx * 6 + 0]; // xx
  dSigma[1] = dSigma_in[idx * 6 + 1]; // xy
  dSigma[2] = dSigma_in[idx * 6 + 2]; // xz
  dSigma[3] = dSigma[1];              // yx
  dSigma[4] = dSigma_in[idx * 6 + 3]; // yy
  dSigma[5] = dSigma_in[idx * 6 + 4]; // yz
  dSigma[6] = dSigma[2];              // zx
  dSigma[7] = dSigma[5];              // zy
  dSigma[8] = dSigma_in[idx * 6 + 5]; // zz

  // dM = 2 * dSigma * M
  float dM[9];
  dM[0] = 2.0f * (dSigma[0] * M[0] + dSigma[1] * M[3] + dSigma[2] * M[6]);
  dM[1] = 2.0f * (dSigma[0] * M[1] + dSigma[1] * M[4] + dSigma[2] * M[7]);
  dM[2] = 2.0f * (dSigma[0] * M[2] + dSigma[1] * M[5] + dSigma[2] * M[8]);
  dM[3] = 2.0f * (dSigma[3] * M[0] + dSigma[4] * M[3] + dSigma[5] * M[6]);
  dM[4] = 2.0f * (dSigma[3] * M[1] + dSigma[4] * M[4] + dSigma[5] * M[7]);
  dM[5] = 2.0f * (dSigma[3] * M[2] + dSigma[4] * M[5] + dSigma[5] * M[8]);
  dM[6] = 2.0f * (dSigma[6] * M[0] + dSigma[7] * M[3] + dSigma[8] * M[6]);
  dM[7] = 2.0f * (dSigma[6] * M[1] + dSigma[7] * M[4] + dSigma[8] * M[7]);
  dM[8] = 2.0f * (dSigma[6] * M[2] + dSigma[7] * M[5] + dSigma[8] * M[8]);

  // dR = dM * S^T (S is diagonal, so S^T=S)
  float dR[9];
  dR[0] = dM[0] * S_x;
  dR[1] = dM[1] * S_y;
  dR[2] = dM[2] * S_z;
  dR[3] = dM[3] * S_x;
  dR[4] = dM[4] * S_y;
  dR[5] = dM[5] * S_z;
  dR[6] = dM[6] * S_x;
  dR[7] = dM[7] * S_y;
  dR[8] = dM[8] * S_z;

  // dS_diag = R^T * dM
  // We only need the diagonal elements for the gradient of scale
  float dS_vec_x = R[0] * dM[0] + R[3] * dM[3] + R[6] * dM[6];
  float dS_vec_y = R[1] * dM[1] + R[4] * dM[4] + R[7] * dM[7];
  float dS_vec_z = R[2] * dM[2] + R[5] * dM[5] + R[8] * dM[8];

  // --- 3. Compute final gradients for scale (s) and quaternion (q) ---

  // Gradient for scale: ds = dS_vec * exp(s)
  dS_in[idx * 3 + 0] = dS_vec_x * S_x;
  dS_in[idx * 3 + 1] = dS_vec_y * S_y;
  dS_in[idx * 3 + 2] = dS_vec_z * S_z;

  // Gradient for quaternion
  // dq_i = sum(dR_jk * dR_jk/dq_i)
  float dw = 0.0f, dx = 0.0f, dy = 0.0f, dz = 0.0f;

  // Row 0
  dw += dR[1] * (-2.0f * z) + dR[2] * (2.0f * y);
  dx += dR[1] * (2.0f * y) + dR[2] * (2.0f * z);
  dy += dR[0] * (-4.0f * y) + dR[1] * (2.0f * x) + dR[2] * (2.0f * w);
  dz += dR[0] * (-4.0f * z) + dR[1] * (-2.0f * w) + dR[2] * (2.0f * x);

  // Row 1
  dw += dR[3] * (2.0f * z) + dR[5] * (-2.0f * x);
  dx += dR[3] * (2.0f * y) + dR[4] * (-4.0f * x) + dR[5] * (-2.0f * w);
  dy += dR[3] * (2.0f * x) + dR[5] * (2.0f * z);
  dz += dR[3] * (2.0f * w) + dR[4] * (-4.0f * z) + dR[5] * (2.0f * y);

  // Row 2
  dw += dR[6] * (-2.0f * y) + dR[7] * (2.0f * x);
  dx += dR[6] * (2.0f * z) + dR[7] * (2.0f * w) + dR[8] * (-4.0f * x);
  dy += dR[6] * (-2.0f * w) + dR[7] * (2.0f * z) + dR[8] * (-4.0f * y);
  dz += dR[6] * (2.0f * x) + dR[7] * (2.0f * y);

  // The raw gradient with respect to the normalized quaternion components (w, x, y, z)
  const float d_norm_q[] = {dw, dx, dy, dz};

  // Calculate the dot product of the normalized quaternion and its gradient
  // This finds the component of the gradient that is parallel to the quaternion vector itself
  const float dot = w * d_norm_q[0] + x * d_norm_q[1] + y * d_norm_q[2] + z * d_norm_q[3];

  // The gradient of the norm is zero for directions orthogonal to the vector.
  // We subtract the parallel component (the projection) and scale by the inverse norm.
  const float inv_norm = 1.0f / norm;
  dQ_in[idx * 4 + 0] = inv_norm * (d_norm_q[0] - dot * w);
  dQ_in[idx * 4 + 1] = inv_norm * (d_norm_q[1] - dot * x);
  dQ_in[idx * 4 + 2] = inv_norm * (d_norm_q[2] - dot * y);
  dQ_in[idx * 4 + 3] = inv_norm * (d_norm_q[3] - dot * z);
}

void compute_sigma_backward(const float *const quaternion, const float *const scale, const float *const sigma_grad_out,
                            const int N, float *quaternion_grad_in, float *scale_grad_in, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(quaternion);
  ASSERT_DEVICE_POINTER(scale);
  ASSERT_DEVICE_POINTER(sigma_grad_out);
  ASSERT_DEVICE_POINTER(quaternion_grad_in);
  ASSERT_DEVICE_POINTER(scale_grad_in);

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  sigma_backward_kernel<<<blocks, threads, 0, stream>>>(quaternion, scale, sigma_grad_out, N, quaternion_grad_in,
                                                        scale_grad_in);
}
