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

__global__ void compute_sigma_backward_kernel(const float *__restrict__ quaternion, const float *__restrict__ scale,
                                              const float *__restrict__ sigma_grad_out, const int N,
                                              float *__restrict__ quaternion_grad_in,
                                              float *__restrict__ scale_grad_in) {
  const int tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + tid;

  if (i >= N)
    return;

  // 1. Load and normalize the quaternion for the current Gaussian
  float w = quaternion[i * 4 + 0], x = quaternion[i * 4 + 1], y = quaternion[i * 4 + 2], z = quaternion[i * 4 + 3];
  const float inv_norm = rsqrtf(w * w + x * x + y * y + z * z);
  w *= inv_norm;
  x *= inv_norm;
  y *= inv_norm;
  z *= inv_norm;

  // 2. Reconstruct the rotation matrix R from the normalized quaternion
  const float R[] = {1.f - 2.f * (y * y + z * z), 2.f * (x * y - w * z),       2.f * (x * z + w * y),
                     2.f * (x * y + w * z),       1.f - 2.f * (x * x + z * z), 2.f * (y * z - w * x),
                     2.f * (x * z - w * y),       2.f * (y * z + w * x),       1.f - 2.f * (x * x + y * y)};

  // 3. Load the log-scale and compute the actual scale factors
  const float sx = __expf(scale[i * 3 + 0]), sy = __expf(scale[i * 3 + 1]), sz = __expf(scale[i * 3 + 2]);

  // 4. Reconstruct the symmetric 3x3 dL/dSigma from the compact 6-element input
  const float dL_dSigma[] = {sigma_grad_out[i * 6 + 0], sigma_grad_out[i * 6 + 1], sigma_grad_out[i * 6 + 2],
                             sigma_grad_out[i * 6 + 1], sigma_grad_out[i * 6 + 3], sigma_grad_out[i * 6 + 4],
                             sigma_grad_out[i * 6 + 2], sigma_grad_out[i * 6 + 4], sigma_grad_out[i * 6 + 5]};

  // BACKWARD PASS (as per the report)
  // --- Step 2.1: Gradient w.r.t. intermediate matrix M ---
  // dL_dM = R^T * dL_dSigma * R
  // First, compute TMP = dL_dSigma * R
  const float TMP[] = {dL_dSigma[0] * R[0] + dL_dSigma[1] * R[3] + dL_dSigma[2] * R[6],
                       dL_dSigma[0] * R[1] + dL_dSigma[1] * R[4] + dL_dSigma[2] * R[7],
                       dL_dSigma[0] * R[2] + dL_dSigma[1] * R[5] + dL_dSigma[2] * R[8],
                       dL_dSigma[3] * R[0] + dL_dSigma[4] * R[3] + dL_dSigma[5] * R[6],
                       dL_dSigma[3] * R[1] + dL_dSigma[4] * R[4] + dL_dSigma[5] * R[7],
                       dL_dSigma[3] * R[2] + dL_dSigma[4] * R[5] + dL_dSigma[5] * R[8],
                       dL_dSigma[6] * R[0] + dL_dSigma[7] * R[3] + dL_dSigma[8] * R[6],
                       dL_dSigma[6] * R[1] + dL_dSigma[7] * R[4] + dL_dSigma[8] * R[7],
                       dL_dSigma[6] * R[2] + dL_dSigma[7] * R[5] + dL_dSigma[8] * R[8]};

  // Then, dL_dM = R^T * TMP
  const float dL_dM[] = {R[0] * TMP[0] + R[3] * TMP[3] + R[6] * TMP[6], R[0] * TMP[1] + R[3] * TMP[4] + R[6] * TMP[7],
                         R[0] * TMP[2] + R[3] * TMP[5] + R[6] * TMP[8], R[1] * TMP[0] + R[4] * TMP[3] + R[7] * TMP[6],
                         R[1] * TMP[1] + R[4] * TMP[4] + R[7] * TMP[7], R[1] * TMP[2] + R[4] * TMP[5] + R[7] * TMP[8],
                         R[2] * TMP[0] + R[5] * TMP[3] + R[8] * TMP[6], R[2] * TMP[1] + R[5] * TMP[4] + R[8] * TMP[7],
                         R[2] * TMP[2] + R[5] * TMP[5] + R[8] * TMP[8]};

  // --- Step 2.2: Gradient w.r.t. scale vector s ---
  // The input 'scale' is log(s), so we compute dL/d(log(s)) = (dL/ds) * s
  // dL/ds = 2 * s * diag(dL/dM) => dL/d(log(s)) = 2 * s^2 * diag(dL/dM)
  scale_grad_in[i * 3 + 0] = 2.f * sx * sx * dL_dM[0];
  scale_grad_in[i * 3 + 1] = 2.f * sy * sy * dL_dM[4];
  scale_grad_in[i * 3 + 2] = 2.f * sz * sz * dL_dM[8];

  // --- Step 2.3: Gradient w.r.t. rotation matrix R ---
  // dL_dR = (dL_dSigma + dL_dSigma^T) * R * M
  // Since dL_dSigma is symmetric, this is 2 * dL_dSigma * R * M
  const float M00 = sx * sx, M11 = sy * sy, M22 = sz * sz;
  const float dL_dR[] = {2.f * TMP[0] * M00, 2.f * TMP[1] * M11, 2.f * TMP[2] * M22,
                         2.f * TMP[3] * M00, 2.f * TMP[4] * M11, 2.f * TMP[5] * M22,
                         2.f * TMP[6] * M00, 2.f * TMP[7] * M11, 2.f * TMP[8] * M22};

  // --- Step 3: Gradient w.r.t. quaternion q ---
  // dL/dqi = Tr((dL/dR)^T * dR/dqi) = sum(dL_dR .* dR_dqi)
  const float dL_dw = dL_dR[1] * (-2.f * z) + dL_dR[2] * (2.f * y) + dL_dR[3] * (2.f * z) + dL_dR[5] * (-2.f * x) +
                      dL_dR[6] * (-2.f * y) + dL_dR[7] * (2.f * x);

  const float dL_dx = dL_dR[1] * (2.f * y) + dL_dR[2] * (2.f * z) + dL_dR[3] * (2.f * y) + dL_dR[4] * (-4.f * x) +
                      dL_dR[5] * (-2.f * w) + dL_dR[6] * (2.f * z) + dL_dR[7] * (2.f * w) + dL_dR[8] * (-4.f * x);

  const float dL_dy = dL_dR[0] * (-4.f * y) + dL_dR[1] * (2.f * x) + dL_dR[2] * (2.f * w) + dL_dR[3] * (2.f * x) +
                      dL_dR[5] * (2.f * z) + dL_dR[6] * (-2.f * w) + dL_dR[7] * (2.f * z) + dL_dR[8] * (-4.f * y);

  const float dL_dz = dL_dR[0] * (-4.f * z) + dL_dR[1] * (-2.f * w) + dL_dR[2] * (2.f * x) + dL_dR[3] * (2.f * w) +
                      dL_dR[4] * (-4.f * z) + dL_dR[6] * (2.f * x) + dL_dR[7] * (2.f * y);

  // Project gradient to be orthogonal to q, and account for normalization chain rule
  const float dot = w * dL_dw + x * dL_dx + y * dL_dy + z * dL_dz;
  quaternion_grad_in[i * 4 + 0] = inv_norm * (dL_dw - w * dot);
  quaternion_grad_in[i * 4 + 1] = inv_norm * (dL_dx - x * dot);
  quaternion_grad_in[i * 4 + 2] = inv_norm * (dL_dy - y * dot);
  quaternion_grad_in[i * 4 + 3] = inv_norm * (dL_dz - z * dot);
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
  compute_sigma_backward_kernel<<<blocks, threads, 0, stream>>>(quaternion, scale, sigma_grad_out, N,
                                                                quaternion_grad_in, scale_grad_in);
}
