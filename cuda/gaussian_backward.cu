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

  const int threads = 1024;
  const int blocks = (N + threads - 1) / threads;
  compute_proj_jacobian_backward_kernel<<<blocks, threads, 0, stream>>>(xyz_c, K, J_grad_out, N, xyz_c_grad_in);
}

__global__ void conic_backward_kernel(const float *__restrict__ J, const float *__restrict__ sigma_world,
                                      const float *__restrict__ camera_T_world,
                                      const float *__restrict__ conic_grad_out, const int N, float *J_grad_in,
                                      float *sigma_world_grad_in) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  // --- 1. Load all inputs into local variables (registers) ---

  const float *J_i = J + i * 6;
  const float *sigma_i = sigma_world + i * 9;

  // Load J (2x3)
  float J00 = J_i[0], J01 = J_i[1], J02 = J_i[2];
  float J10 = J_i[3], J11 = J_i[4], J12 = J_i[5];

  // Load sigma_world (3x3)
  float S00 = sigma_i[0], S01 = sigma_i[1], S02 = sigma_i[2];
  float S10 = sigma_i[3], S11 = sigma_i[4], S12 = sigma_i[5];
  float S20 = sigma_i[6], S21 = sigma_i[7], S22 = sigma_i[8];

  // Load W (3x3 rotation matrix)
  float W00 = camera_T_world[0], W01 = camera_T_world[1], W02 = camera_T_world[2];
  float W10 = camera_T_world[4], W11 = camera_T_world[5], W12 = camera_T_world[6];
  float W20 = camera_T_world[8], W21 = camera_T_world[9], W22 = camera_T_world[10];

  // Load and reconstruct symmetric grad_sigma_image (2x2)
  float G00 = conic_grad_out[i * 3 + 0];
  float G01 = conic_grad_out[i * 3 + 1];
  float G11 = conic_grad_out[i * 3 + 2];
  float G10 = G01; // Symmetry

  // --- 2. Compute intermediate products using registers ---

  // JW = J @ W (2x3 @ 3x3 -> 2x3)
  float JW00 = J00 * W00 + J01 * W10 + J02 * W20;
  float JW01 = J00 * W01 + J01 * W11 + J02 * W21;
  float JW02 = J00 * W02 + J01 * W12 + J02 * W22;
  float JW10 = J10 * W00 + J11 * W10 + J12 * W20;
  float JW11 = J10 * W01 + J11 * W11 + J12 * W21;
  float JW12 = J10 * W02 + J11 * W12 + J12 * W22;

  // V = grad_sigma_image @ JW (2x2 @ 2x3 -> 2x3)
  float V00 = G00 * JW00 + G01 * JW10;
  float V01 = G00 * JW01 + G01 * JW11;
  float V02 = G00 * JW02 + G01 * JW12;
  float V10 = G10 * JW00 + G11 * JW10;
  float V11 = G10 * JW01 + G11 * JW11;
  float V12 = G10 * JW02 + G11 * JW12;

  // --- 3. Compute and write output gradients ---

  // A. Gradient w.r.t. sigma_world = JW.T @ V (3x2 @ 2x3 -> 3x3)
  float *out_sigma_grad = sigma_world_grad_in + i * 9;
  // Since d(sigma_world) is symmetric, we compute the full matrix product
  // and then can optionally just store the upper/lower triangular part if
  // the next kernel expects that. Here we compute the full 3x3 matrix.
  float grad_S00 = JW00 * V00 + JW10 * V10;
  float grad_S01 = JW00 * V01 + JW10 * V11;
  float grad_S02 = JW00 * V02 + JW10 * V12;
  float grad_S10 = JW01 * V00 + JW11 * V10;
  float grad_S11 = JW01 * V01 + JW11 * V11;
  float grad_S12 = JW01 * V02 + JW11 * V12;
  float grad_S20 = JW02 * V00 + JW12 * V10;
  float grad_S21 = JW02 * V01 + JW12 * V11;
  float grad_S22 = JW02 * V02 + JW12 * V12;
  // Store the full symmetric gradient
  out_sigma_grad[0] = grad_S00;
  out_sigma_grad[1] = (grad_S01 + grad_S10) * 0.5f;
  out_sigma_grad[2] = (grad_S02 + grad_S20) * 0.5f;
  out_sigma_grad[3] = out_sigma_grad[1]; // yx = xy
  out_sigma_grad[4] = grad_S11;
  out_sigma_grad[5] = (grad_S12 + grad_S21) * 0.5f;
  out_sigma_grad[6] = out_sigma_grad[2]; // zx = xz
  out_sigma_grad[7] = out_sigma_grad[5]; // zy = yz
  out_sigma_grad[8] = grad_S22;

  // B. Gradient w.r.t. J = 2 * (V @ sigma_world @ W.T)
  // Step B1: V_sigma = V @ sigma_world (2x3 @ 3x3 -> 2x3)
  float VS00 = V00 * S00 + V01 * S10 + V02 * S20;
  float VS01 = V00 * S01 + V01 * S11 + V02 * S21;
  float VS02 = V00 * S02 + V01 * S12 + V02 * S22;
  float VS10 = V10 * S00 + V11 * S10 + V12 * S20;
  float VS11 = V10 * S01 + V11 * S11 + V12 * S21;
  float VS12 = V10 * S02 + V11 * S12 + V12 * S22;

  // Step B2: J_grad = V_sigma @ W.T (2x3 @ 3x3 -> 2x3), then scale by 2
  float *out_J_grad = J_grad_in + i * 6;
  out_J_grad[0] = (VS00 * W00 + VS01 * W01 + VS02 * W02) * 2.0f;
  out_J_grad[1] = (VS00 * W10 + VS01 * W11 + VS02 * W12) * 2.0f;
  out_J_grad[2] = (VS00 * W20 + VS01 * W21 + VS02 * W22) * 2.0f;
  out_J_grad[3] = (VS10 * W00 + VS11 * W01 + VS12 * W02) * 2.0f;
  out_J_grad[4] = (VS10 * W10 + VS11 * W11 + VS12 * W12) * 2.0f;
  out_J_grad[5] = (VS10 * W20 + VS11 * W21 + VS12 * W22) * 2.0f;
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

  const int threads = 1024;
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

  const int threads = 1024;
  const int blocks = (N + threads - 1) / threads;
  sigma_backward_kernel<<<blocks, threads, 0, stream>>>(quaternion, scale, sigma_grad_out, N, quaternion_grad_in,
                                                        scale_grad_in);
}
