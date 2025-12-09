// spherical_harmonics_backward.cu

#include "checks.cuh"
#include "gsplat_cuda/cuda_backward.cuh"
#include "sphericart_cuda.hpp"
#include <thrust/device_vector.h>

__global__ void compute_sh_gradients_kernel(const float *d_sph, const float *d_dsph, const float *d_rgb_vals,
                                            const float *d_sh_coeffs, const float *rgb_grad_out, const int n_coeffs,
                                            const int N, float *sh_grad_in, float *sh_grad_band_0_in,
                                            float *xyz_c_grad_in) {
  // Determine the unique index for this thread, corresponding to a single point/Gaussian.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  // Set up pointers to the data for the current point.
  const float *point_sph_vals = d_sph + idx * n_coeffs;
  const float *point_dsph_vals = d_dsph + idx * n_coeffs * 3;
  // Band 0 coeffs (RGB) are stored separately
  const float *point_rgb_vals = d_rgb_vals + idx * 3;

  // Higher order SH coeffs.
  // Note: if n_coeffs > 1, sh_coeffs stores (n_coeffs - 1) * 3 floats per gaussian.
  const float *point_sh_coeffs = nullptr;
  if (n_coeffs > 1) {
    point_sh_coeffs = d_sh_coeffs + idx * (n_coeffs - 1) * 3;
  }

  const float *point_rgb_grad = rgb_grad_out + idx * 3;
  // Pointer for the new band 0 gradient output
  float *point_sh_grad_band_0 = sh_grad_band_0_in + idx * 3;

  float dR_dx = 0.0f;
  float dG_dx = 0.0f;
  float dB_dx = 0.0f;
  float dR_dy = 0.0f;
  float dG_dy = 0.0f;
  float dB_dy = 0.0f;
  float dR_dz = 0.0f;
  float dG_dz = 0.0f;
  float dB_dz = 0.0f;

  // --- Gradient for Band 0 Coefficients ---
  // The gradient for the band 0 coefficient is simply the incoming gradient
  // from the logit, as its derivative in the forward pass is 1.
  point_sh_grad_band_0[0] = point_rgb_grad[0] * point_sph_vals[0];
  point_sh_grad_band_0[1] = point_rgb_grad[1] * point_sph_vals[0];
  point_sh_grad_band_0[2] = point_rgb_grad[2] * point_sph_vals[0];

  // d_dsph layout: [n_coeffs, 3] (x, y, z)
  // idx * n_coeffs * 3 + i * 3 + axis
  float d_Y0_dx = point_dsph_vals[0 * 3 + 0];
  float d_Y0_dy = point_dsph_vals[0 * 3 + 1];
  float d_Y0_dz = point_dsph_vals[0 * 3 + 2];

  // Band 0 coeffs
  float R0 = point_rgb_vals[0];
  float G0 = point_rgb_vals[1];
  float B0 = point_rgb_vals[2];

  dR_dx += d_Y0_dx * R0;
  dG_dx += d_Y0_dx * G0;
  dB_dx += d_Y0_dx * B0;

  dR_dy += d_Y0_dy * R0;
  dG_dy += d_Y0_dy * G0;
  dB_dy += d_Y0_dy * B0;

  dR_dz += d_Y0_dz * R0;
  dG_dz += d_Y0_dz * G0;
  dB_dz += d_Y0_dz * B0;

  // --- Gradients for Higher-Order Coefficients (l > 0) ---
  // The chain rule: dL/d(coeff) = dL/d(logit) * d(logit)/d(coeff).
  // For higher-order bands, d(logit)/d(coeff) is the corresponding sh_val.
  if (n_coeffs > 1) {
    float *point_sh_grad = sh_grad_in + idx * (n_coeffs - 1) * 3; // subtract 1 to account for band 0
    for (int i = 0; i < n_coeffs - 1; ++i) {                      // Start loop from 1
      float sh_val = point_sph_vals[i + 1];
      point_sh_grad[i * 3 + 0] = point_rgb_grad[0] * sh_val; // Gradient for Red
      point_sh_grad[i * 3 + 1] = point_rgb_grad[1] * sh_val; // Gradient for Green
      point_sh_grad[i * 3 + 2] = point_rgb_grad[2] * sh_val; // Gradient for Blue

      // Gradient w.r.t Position
      int coeff_idx = i + 1;
      float d_Yi_dx = point_dsph_vals[coeff_idx * 3 + 0];
      float d_Yi_dy = point_dsph_vals[coeff_idx * 3 + 1];
      float d_Yi_dz = point_dsph_vals[coeff_idx * 3 + 2];

      float Ri = point_sh_coeffs[i * 3 + 0];
      float Gi = point_sh_coeffs[i * 3 + 1];
      float Bi = point_sh_coeffs[i * 3 + 2];

      dR_dx += d_Yi_dx * Ri;
      dG_dx += d_Yi_dx * Gi;
      dB_dx += d_Yi_dx * Bi;

      dR_dy += d_Yi_dy * Ri;
      dG_dy += d_Yi_dy * Gi;
      dB_dy += d_Yi_dy * Bi;

      dR_dz += d_Yi_dz * Ri;
      dG_dz += d_Yi_dz * Gi;
      dB_dz += d_Yi_dz * Bi;
    }
  }

  // Accumulate total gradient w.r.t xyz_c
  // dL/d(xyz) = dL/dR * dR/d(xyz) + dL/dG * dG/d(xyz) + dL/dB * dB/d(xyz)
  float total_grad_x = point_rgb_grad[0] * dR_dx + point_rgb_grad[1] * dG_dx + point_rgb_grad[2] * dB_dx;
  float total_grad_y = point_rgb_grad[0] * dR_dy + point_rgb_grad[1] * dG_dy + point_rgb_grad[2] * dB_dy;
  float total_grad_z = point_rgb_grad[0] * dR_dz + point_rgb_grad[1] * dG_dz + point_rgb_grad[2] * dB_dz;

  xyz_c_grad_in[idx * 3 + 0] += total_grad_x;
  xyz_c_grad_in[idx * 3 + 1] += total_grad_y;
  xyz_c_grad_in[idx * 3 + 2] += total_grad_z;
}

void precompute_spherical_harmonics_backward(const float *const xyz_c, const float *const rgb_vals,
                                             const float *const sh_coeffs, const float *const rgb_grad_out,
                                             const int l_max, const int N, float *sh_grad_in, float *sh_grad_band_0_in,
                                             float *xyz_c_grad_in, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(xyz_c);
  ASSERT_DEVICE_POINTER(rgb_vals);
  if (l_max > 0)
    ASSERT_DEVICE_POINTER(sh_coeffs);
  ASSERT_DEVICE_POINTER(rgb_grad_out);
  ASSERT_DEVICE_POINTER(sh_grad_band_0_in);
  if (l_max > 0)
    ASSERT_DEVICE_POINTER(sh_grad_in);
  ASSERT_DEVICE_POINTER(xyz_c_grad_in);

  // Initialize the sphericart calculator for the given maximum degree.
  sphericart::cuda::SphericalHarmonics<float> calculator_cuda(l_max);

  const int n_coeffs = (l_max + 1) * (l_max + 1);

  // Allocate temporary device memory for SH basis values (d_sph) and their derivatives
  // with respect to Cartesian coordinates (d_dsph) using thrust::device_vector.
  // Memory is automatically allocated here.
  thrust::device_vector<float> d_sph(N * n_coeffs);
  thrust::device_vector<float> d_dsph(N * n_coeffs * 3);

  // Use the sphericart library to compute the SH basis values.
  // We pass the raw pointers from the device_vectors.
  calculator_cuda.compute_with_gradients(xyz_c, N, thrust::raw_pointer_cast(d_sph.data()),
                                         thrust::raw_pointer_cast(d_dsph.data()), stream);

  // Define CUDA kernel launch parameters.
  const int blockSize = 256;
  const int gridSize = (N + blockSize - 1) / blockSize;

  // Launch the kernel to compute the final SH coefficient gradients.
  compute_sh_gradients_kernel<<<gridSize, blockSize, 0, stream>>>(
      thrust::raw_pointer_cast(d_sph.data()), thrust::raw_pointer_cast(d_dsph.data()), rgb_vals, sh_coeffs,
      rgb_grad_out, n_coeffs, N, sh_grad_in, sh_grad_band_0_in, xyz_c_grad_in);
}
