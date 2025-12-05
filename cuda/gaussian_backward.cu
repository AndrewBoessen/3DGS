// projection_backward.cu

#include "checks.cuh"
#include "gsplat_cuda/cuda_backward.cuh"

__global__ void compute_projection_jacobian_backward_kernel(const float *__restrict__ xyz,
                                                            const float *__restrict__ proj,
                                                            const float *__restrict__ J_grad_out, const int N,
                                                            float *xyz_grad_in) {
  constexpr int XYZ_STRIDE = 3;
  constexpr int J_STRIDE = 6;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = threadIdx.x & 0x1f;

  // load and broadcast Proj to all threads in warp
  float p_val = 0.0f;
  if (lane_id < 16) {
    p_val = proj[lane_id];
  }
  const float p00 = __shfl_sync(0xffffffff, p_val, 0);
  const float p01 = __shfl_sync(0xffffffff, p_val, 1);
  const float p02 = __shfl_sync(0xffffffff, p_val, 2);
  const float p03 = __shfl_sync(0xffffffff, p_val, 3);
  const float p10 = __shfl_sync(0xffffffff, p_val, 4);
  const float p11 = __shfl_sync(0xffffffff, p_val, 5);
  const float p12 = __shfl_sync(0xffffffff, p_val, 6);
  const float p13 = __shfl_sync(0xffffffff, p_val, 7);
  const float p30 = __shfl_sync(0xffffffff, p_val, 12);
  const float p31 = __shfl_sync(0xffffffff, p_val, 13);
  const float p32 = __shfl_sync(0xffffffff, p_val, 14);
  const float p33 = __shfl_sync(0xffffffff, p_val, 15);

  if (i >= N) {
    return;
  }

  float x = xyz[i * XYZ_STRIDE + 0];
  float y = xyz[i * XYZ_STRIDE + 1];
  float z = xyz[i * XYZ_STRIDE + 2];

  // Clip coordinates
  float xc = p00 * x + p01 * y + p02 * z + p03;
  float yc = p10 * x + p11 * y + p12 * z + p13;
  float wc = p30 * x + p31 * y + p32 * z + p33;

  if (fabsf(wc) < 1e-6f) {
    return;
  }

  float wc_inv = 1.0f / wc;
  float wc_inv2 = wc_inv * wc_inv;
  float wc_inv3 = wc_inv2 * wc_inv;

  // Gradients of J
  float dJ_00 = J_grad_out[i * J_STRIDE + 0];
  float dJ_01 = J_grad_out[i * J_STRIDE + 1];
  float dJ_02 = J_grad_out[i * J_STRIDE + 2];
  float dJ_10 = J_grad_out[i * J_STRIDE + 3];
  float dJ_11 = J_grad_out[i * J_STRIDE + 4];
  float dJ_12 = J_grad_out[i * J_STRIDE + 5];

  // Backprop through J calculation
  // J00 = (p00*wc - xc*p30) / wc^2
  // Let Num00 = p00*wc - xc*p30
  // J00 = Num00 * wc^-2
  // dNum00 = dJ00 * wc^-2
  // dwc += dJ00 * Num00 * (-2 * wc^-3) = dJ00 * J00 * (-2/wc)
  // But we don't have J00 computed here.
  // Alternatively:
  // d(J00)/d(xc) = -p30 / wc^2
  // d(J00)/d(wc) = (p00 * wc^2 - (p00*wc - xc*p30) * 2*wc) / wc^4
  //              = (p00*wc - 2*(p00*wc - xc*p30)) / wc^3
  //              = (p00*wc - 2*p00*wc + 2*xc*p30) / wc^3
  //              = (2*xc*p30 - p00*wc) / wc^3

  float dxc = 0.0f;
  float dyc = 0.0f;
  float dwc = 0.0f;

  // Row 0
  // J00
  dxc += dJ_00 * (-p30 * wc_inv2);
  dwc += dJ_00 * (2.0f * xc * p30 - p00 * wc) * wc_inv3;
  // J01
  dxc += dJ_01 * (-p31 * wc_inv2);
  dwc += dJ_01 * (2.0f * xc * p31 - p01 * wc) * wc_inv3;
  // J02
  dxc += dJ_02 * (-p32 * wc_inv2);
  dwc += dJ_02 * (2.0f * xc * p32 - p02 * wc) * wc_inv3;

  // Row 1
  // J10
  dyc += dJ_10 * (-p30 * wc_inv2);
  dwc += dJ_10 * (2.0f * yc * p30 - p10 * wc) * wc_inv3;
  // J11
  dyc += dJ_11 * (-p31 * wc_inv2);
  dwc += dJ_11 * (2.0f * yc * p31 - p11 * wc) * wc_inv3;
  // J12
  dyc += dJ_12 * (-p32 * wc_inv2);
  dwc += dJ_12 * (2.0f * yc * p32 - p12 * wc) * wc_inv3;

  // Backprop from Clip to Camera
  // xc = p00*x + p01*y + p02*z + p03
  // yc = p10*x + p11*y + p12*z + p13
  // wc = p30*x + p31*y + p32*z + p33

  float dx = dxc * p00 + dyc * p10 + dwc * p30;
  float dy = dxc * p01 + dyc * p11 + dwc * p31;
  float dz = dxc * p02 + dyc * p12 + dwc * p32;

  xyz_grad_in[i * XYZ_STRIDE + 0] += dx;
  xyz_grad_in[i * XYZ_STRIDE + 1] += dy;
  xyz_grad_in[i * XYZ_STRIDE + 2] += dz;
}

void compute_projection_jacobian_backward(const float *const xyz_c, const float *const proj,
                                          const float *const J_grad_out, const int N, float *xyz_c_grad_in,
                                          cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(xyz_c);
  ASSERT_DEVICE_POINTER(proj);
  ASSERT_DEVICE_POINTER(J_grad_out);
  ASSERT_DEVICE_POINTER(xyz_c_grad_in);

  const int threads_per_block = 256;
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  compute_projection_jacobian_backward_kernel<<<gridsize, blocksize, 0, stream>>>(xyz_c, proj, J_grad_out, N,
                                                                                  xyz_c_grad_in);
}

__global__ void conic_backward_kernel(const float *__restrict__ J, const float *__restrict__ sigma,
                                      const float *__restrict__ view, const float *__restrict__ conic,
                                      const float *__restrict__ conic_grad_out, const int N, float *J_grad_in,
                                      float *sigma_grad_in) {
  constexpr int SIGMA_STRIDE = 6;
  constexpr int J_STRIDE = 6;
  constexpr int CONIC_STRIDE = 3;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = threadIdx.x & 0x1f;

  // Load and broadcast View Matrix (4x4) within warp
  float v_val = 0.0f;
  if (lane_id < 16) {
    v_val = view[lane_id];
  }
  // W (rotation part) = [r00, r01, r02, r10, r11, r12, r20, r21, r22]
  const float w00 = __shfl_sync(0xffffffff, v_val, 0);
  const float w01 = __shfl_sync(0xffffffff, v_val, 1);
  const float w02 = __shfl_sync(0xffffffff, v_val, 2);
  const float w10 = __shfl_sync(0xffffffff, v_val, 4);
  const float w11 = __shfl_sync(0xffffffff, v_val, 5);
  const float w12 = __shfl_sync(0xffffffff, v_val, 6);
  const float w20 = __shfl_sync(0xffffffff, v_val, 8);
  const float w21 = __shfl_sync(0xffffffff, v_val, 9);
  const float w22 = __shfl_sync(0xffffffff, v_val, 10);

  if (i >= N) {
    return;
  }

  // Load J
  const int j_base_idx = i * J_STRIDE;
  const float j00 = J[j_base_idx + 0];
  const float j01 = J[j_base_idx + 1];
  const float j02 = J[j_base_idx + 2];
  const float j10 = J[j_base_idx + 3];
  const float j11 = J[j_base_idx + 4];
  const float j12 = J[j_base_idx + 5];

  // Load Sigma (symmetric)
  const int sigma_base_idx = i * SIGMA_STRIDE;
  const float s00 = sigma[sigma_base_idx + 0];
  const float s01 = sigma[sigma_base_idx + 1];
  const float s02 = sigma[sigma_base_idx + 2];
  const float s11 = sigma[sigma_base_idx + 3];
  const float s12 = sigma[sigma_base_idx + 4];
  const float s22 = sigma[sigma_base_idx + 5];

  // Recompute M = J @ W
  const float m00 = j00 * w00 + j01 * w10 + j02 * w20;
  const float m01 = j00 * w01 + j01 * w11 + j02 * w21;
  const float m02 = j00 * w02 + j01 * w12 + j02 * w22;
  const float m10 = j10 * w00 + j11 * w10 + j12 * w20;
  const float m11 = j10 * w01 + j11 * w11 + j12 * w21;
  const float m12 = j10 * w02 + j11 * w12 + j12 * w22;

  // Recompute V = Sigma @ M^T
  const float v00 = s00 * m00 + s01 * m01 + s02 * m02;
  const float v01 = s00 * m10 + s01 * m11 + s02 * m12;
  const float v10 = s01 * m00 + s11 * m01 + s12 * m02;
  const float v11 = s01 * m10 + s11 * m11 + s12 * m12;
  const float v20 = s02 * m00 + s12 * m01 + s22 * m02;
  const float v21 = s02 * m10 + s12 * m11 + s22 * m12;

  // Load gradients for Conic (dC)
  const int conic_base_idx = i * CONIC_STRIDE;
  const float dc00_out = conic_grad_out[conic_base_idx + 0];
  const float dc01_out = conic_grad_out[conic_base_idx + 1];
  const float dc11_out = conic_grad_out[conic_base_idx + 2];

  // Load Conic (C) - inverse covariance
  const float c00 = conic[conic_base_idx + 0];
  const float c01 = conic[conic_base_idx + 1];
  const float c11 = conic[conic_base_idx + 2];

  // Compute dSigma_prime = - C * dC * C
  // T = C * dC
  const float t00 = c00 * dc00_out + c01 * dc01_out;
  const float t01 = c00 * dc01_out + c01 * dc11_out;
  const float t10 = c01 * dc00_out + c11 * dc01_out;
  const float t11 = c01 * dc01_out + c11 * dc11_out;

  // dS = - T * C
  const float d_c00 = -(t00 * c00 + t01 * c01);
  const float d_c01 = -(t00 * c01 + t01 * c11);
  // const float d_c10 = -(t10 * c00 + t11 * c01); // Should be same as d_c01
  const float d_c11 = -(t10 * c01 + t11 * c11);

  // Backprop Conic = M @ V
  // c00 = m00*v00 + m01*v10 + m02*v20
  // c01 = m00*v01 + m01*v11 + m02*v21
  // c11 = m10*v01 + m11*v11 + m12*v21

  // Compute dL/dV
  float dv00 = d_c00 * m00 + d_c01 * m10;
  float dv01 = d_c01 * m00 + d_c11 * m10;
  float dv10 = d_c00 * m01 + d_c01 * m11;
  float dv11 = d_c01 * m01 + d_c11 * m11;
  float dv20 = d_c00 * m02 + d_c01 * m12;
  float dv21 = d_c01 * m02 + d_c11 * m12;

  // Compute dL/dSigma = dL/dV @ M
  // Note: sigma_grad_in is symmetric, so we sum contributions for s_ij and s_ji
  float ds00 = dv00 * m00 + dv01 * m10;
  float ds01 = dv00 * m01 + dv01 * m11 + dv10 * m00 + dv11 * m10; // s01 and s10
  float ds02 = dv00 * m02 + dv01 * m12 + dv20 * m00 + dv21 * m10; // s02 and s20
  float ds11 = dv10 * m01 + dv11 * m11;
  float ds12 = dv10 * m02 + dv11 * m12 + dv20 * m01 + dv21 * m11; // s12 and s21
  float ds22 = dv20 * m02 + dv21 * m12;

  sigma_grad_in[sigma_base_idx + 0] += ds00;
  sigma_grad_in[sigma_base_idx + 1] += ds01; // Store upper triangle, sum contributions
  sigma_grad_in[sigma_base_idx + 2] += ds02;
  sigma_grad_in[sigma_base_idx + 3] += ds11;
  sigma_grad_in[sigma_base_idx + 4] += ds12;
  sigma_grad_in[sigma_base_idx + 5] += ds22;

  // Compute dL/dM (from Conic)
  float dm_from_conic_00 = d_c00 * v00 + d_c01 * v01;
  float dm_from_conic_01 = d_c00 * v10 + d_c01 * v11;
  float dm_from_conic_02 = d_c00 * v20 + d_c01 * v21;
  float dm_from_conic_10 = d_c01 * v00 + d_c11 * v01;
  float dm_from_conic_11 = d_c01 * v10 + d_c11 * v11;
  float dm_from_conic_12 = d_c01 * v20 + d_c11 * v21;

  // Compute dL/dM (from V = Sigma @ M^T) = (dL/dV)^T @ Sigma
  float dm_from_V_00 = dv00 * s00 + dv10 * s01 + dv20 * s02;
  float dm_from_V_01 = dv00 * s01 + dv10 * s11 + dv20 * s12;
  float dm_from_V_02 = dv00 * s02 + dv10 * s12 + dv20 * s22;
  float dm_from_V_10 = dv01 * s00 + dv11 * s01 + dv21 * s02;
  float dm_from_V_11 = dv01 * s01 + dv11 * s11 + dv21 * s12;
  float dm_from_V_12 = dv01 * s02 + dv11 * s12 + dv21 * s22;

  // Total dL/dM
  float dm00 = dm_from_conic_00 + dm_from_V_00;
  float dm01 = dm_from_conic_01 + dm_from_V_01;
  float dm02 = dm_from_conic_02 + dm_from_V_02;
  float dm10 = dm_from_conic_10 + dm_from_V_10;
  float dm11 = dm_from_conic_11 + dm_from_V_11;
  float dm12 = dm_from_conic_12 + dm_from_V_12;

  // Compute dL/dJ = dL/dM @ W^T
  J_grad_in[j_base_idx + 0] += dm00 * w00 + dm01 * w01 + dm02 * w02;
  J_grad_in[j_base_idx + 1] += dm00 * w10 + dm01 * w11 + dm02 * w12;
  J_grad_in[j_base_idx + 2] += dm00 * w20 + dm01 * w21 + dm02 * w22;
  J_grad_in[j_base_idx + 3] += dm10 * w00 + dm11 * w01 + dm12 * w02;
  J_grad_in[j_base_idx + 4] += dm10 * w10 + dm11 * w11 + dm12 * w12;
  J_grad_in[j_base_idx + 5] += dm10 * w20 + dm11 * w21 + dm12 * w22;
}

void compute_conic_backward(const float *const J, const float *const sigma, const float *const view,
                            const float *const conic, const float *const conic_grad_out, const int N, float *J_grad_in,
                            float *sigma_grad_in, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(J);
  ASSERT_DEVICE_POINTER(sigma);
  ASSERT_DEVICE_POINTER(view);
  ASSERT_DEVICE_POINTER(conic);
  ASSERT_DEVICE_POINTER(conic_grad_out);
  ASSERT_DEVICE_POINTER(J_grad_in);
  ASSERT_DEVICE_POINTER(sigma_grad_in);

  const int threads_per_block = 256;
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  conic_backward_kernel<<<gridsize, blocksize, 0, stream>>>(J, sigma, view, conic, conic_grad_out, N, J_grad_in,
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
  const float norm = sqrtf(qw * qw + qx * qx + qy * qy + qz * qz);
  const float inv_norm = 1.0f / (norm + 1e-6f);
  const float w = qw * inv_norm;
  const float x = qx * inv_norm;
  const float y = qy * inv_norm;
  const float z = qz * inv_norm;

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
  dSigma[3] = dSigma_in[idx * 6 + 1]; // yx = xy
  dSigma[4] = dSigma_in[idx * 6 + 3]; // yy
  dSigma[5] = dSigma_in[idx * 6 + 4]; // yz
  dSigma[6] = dSigma_in[idx * 6 + 2]; // zx = xz
  dSigma[7] = dSigma_in[idx * 6 + 4]; // zy = yz
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
