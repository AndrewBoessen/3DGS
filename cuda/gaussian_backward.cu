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

__global__ void conic_backward_kernel(const float *const J, const float *const sigma, const float *const T,
                                      const float *const conic_grad_out, const int N, float *J_grad_in,
                                      float *sigma_grad_in) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  // --- Load Data ---

  // Load camera rotation matrix W (top-left 3x3 of T) into registers.
  // This is the same for all threads and will be cached.
  // W = | W0 W1 W2 |
  //     | W3 W4 W5 |
  //     | W6 W7 W8 |
  float W0 = T[0], W1 = T[1], W2 = T[2];
  float W3 = T[4], W4 = T[5], W5 = T[6];
  float W6 = T[8], W7 = T[9], W8 = T[10];

  // Pointers to the data for the current Gaussian
  const float *jacobian = J + idx * 6;
  const float *cov3D = sigma + idx * 6;
  const float *conic_grad = conic_grad_out + idx * 3;

  // Load Jacobian J
  float J0 = jacobian[0], J1 = jacobian[1], J2 = jacobian[2];
  float J3 = jacobian[3], J4 = jacobian[4], J5 = jacobian[5];

  // Load and reconstruct full 3x3 symmetric world-space covariance Sigma (S)
  float S0 = cov3D[0], S1 = cov3D[1], S2 = cov3D[2];
  float S3 = cov3D[3], S4 = cov3D[4], S5 = cov3D[5];
  float S10 = S1, S20 = S2, S21 = S4;

  // Load and reconstruct full 2x2 symmetric conic gradient Gc (dL/dC)
  float Gc0 = conic_grad[0], Gc1 = conic_grad[1], Gc2 = conic_grad[2];
  float Gc10 = Gc1;

  // --- Compute dL/dJ = 2 * Gc * J * (W * S * W^T) ---

  // 1. Compute Temp_WS = W * S (3x3)
  float WS00 = W0 * S0 + W1 * S10 + W2 * S20;
  float WS01 = W0 * S1 + W1 * S3 + W2 * S21;
  float WS02 = W0 * S2 + W1 * S4 + W2 * S5;
  float WS10 = W3 * S0 + W4 * S10 + W5 * S20;
  float WS11 = W3 * S1 + W4 * S3 + W5 * S21;
  float WS12 = W3 * S2 + W4 * S4 + W5 * S5;
  float WS20 = W6 * S0 + W7 * S10 + W8 * S20;
  float WS21 = W6 * S1 + W7 * S3 + W8 * S21;
  float WS22 = W6 * S2 + W7 * S4 + W8 * S5;

  // 2. Compute Sigma_cam (Sc) = Temp_WS * W^T (3x3, symmetric)
  float Sc00 = WS00 * W0 + WS01 * W1 + WS02 * W2;
  float Sc11 = WS10 * W3 + WS11 * W4 + WS12 * W5;
  float Sc22 = WS20 * W6 + WS21 * W7 + WS22 * W8;
  float Sc01 = WS00 * W3 + WS01 * W4 + WS02 * W5;
  float Sc02 = WS00 * W6 + WS01 * W7 + WS02 * W8;
  float Sc12 = WS10 * W6 + WS11 * W7 + WS12 * W8;
  float Sc10 = Sc01, Sc20 = Sc02, Sc21 = Sc12;

  // 3. Compute M = J * Sigma_cam (2x3)
  float M0 = J0 * Sc00 + J1 * Sc10 + J2 * Sc20;
  float M1 = J0 * Sc01 + J1 * Sc11 + J2 * Sc21;
  float M2 = J0 * Sc02 + J1 * Sc12 + J2 * Sc22;
  float M3 = J3 * Sc00 + J4 * Sc10 + J5 * Sc20;
  float M4 = J3 * Sc01 + J4 * Sc11 + J5 * Sc21;
  float M5 = J3 * Sc02 + J4 * Sc12 + J5 * Sc22;

  // 4. Compute dL/dJ = 2 * Gc * M (2x3) and store
  J_grad_in[idx * 6 + 0] = 2.0f * (Gc0 * M0 + Gc1 * M3);
  J_grad_in[idx * 6 + 1] = 2.0f * (Gc0 * M1 + Gc1 * M4);
  J_grad_in[idx * 6 + 2] = 2.0f * (Gc0 * M2 + Gc1 * M5);
  J_grad_in[idx * 6 + 3] = 2.0f * (Gc10 * M0 + Gc2 * M3);
  J_grad_in[idx * 6 + 4] = 2.0f * (Gc10 * M1 + Gc2 * M4);
  J_grad_in[idx * 6 + 5] = 2.0f * (Gc10 * M2 + Gc2 * M5);

  // --- Compute dL/dSigma = W^T * (J^T * Gc * J) * W ---

  // 1. Compute V = Gc * J (2x3)
  float V0 = Gc0 * J0 + Gc1 * J3;
  float V1 = Gc0 * J1 + Gc1 * J4;
  float V2 = Gc0 * J2 + Gc1 * J5;
  float V3 = Gc10 * J0 + Gc2 * J3;
  float V4 = Gc10 * J1 + Gc2 * J4;
  float V5 = Gc10 * J2 + Gc2 * J5;

  // 2. Compute G_Scam = J^T * V (3x3, symmetric)
  float Gsc00 = J0 * V0 + J3 * V3;
  float Gsc11 = J1 * V1 + J4 * V4;
  float Gsc22 = J2 * V2 + J5 * V5;
  float Gsc01 = J0 * V1 + J3 * V4;
  float Gsc02 = J0 * V2 + J3 * V5;
  float Gsc12 = J1 * V2 + J4 * V5;
  float Gsc10 = Gsc01, Gsc20 = Gsc02, Gsc21 = Gsc12;

  // 3. Compute Temp_WtG = W^T * G_Scam (3x3)
  float WtG00 = W0 * Gsc00 + W3 * Gsc10 + W6 * Gsc20;
  float WtG01 = W0 * Gsc01 + W3 * Gsc11 + W6 * Gsc21;
  float WtG02 = W0 * Gsc02 + W3 * Gsc12 + W6 * Gsc22;
  float WtG10 = W1 * Gsc00 + W4 * Gsc10 + W7 * Gsc20;
  float WtG11 = W1 * Gsc01 + W4 * Gsc11 + W7 * Gsc21;
  float WtG12 = W1 * Gsc02 + W4 * Gsc12 + W7 * Gsc22;
  float WtG20 = W2 * Gsc00 + W5 * Gsc10 + W8 * Gsc20;
  float WtG21 = W2 * Gsc01 + W5 * Gsc11 + W8 * Gsc21;
  float WtG22 = W2 * Gsc02 + W5 * Gsc12 + W8 * Gsc22;

  // 4. Compute dL/dSigma = Temp_WtG * W (3x3, symmetric) and store compact form
  sigma_grad_in[idx * 6 + 0] = WtG00 * W0 + WtG01 * W1 + WtG02 * W2; // (0,0)
  sigma_grad_in[idx * 6 + 1] = WtG00 * W3 + WtG01 * W4 + WtG02 * W5; // (0,1)
  sigma_grad_in[idx * 6 + 2] = WtG00 * W6 + WtG01 * W7 + WtG02 * W8; // (0,2)
  sigma_grad_in[idx * 6 + 3] = WtG10 * W3 + WtG11 * W4 + WtG12 * W5; // (1,1)
  sigma_grad_in[idx * 6 + 4] = WtG10 * W6 + WtG11 * W7 + WtG12 * W8; // (1,2)
  sigma_grad_in[idx * 6 + 5] = WtG20 * W6 + WtG21 * W7 + WtG22 * W8; // (2,2)
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
  conic_backward_kernel<<<blocks, threads, 0, stream>>>(J, sigma, T, conic_grad_out, N, J_grad_in, sigma_grad_in);
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
