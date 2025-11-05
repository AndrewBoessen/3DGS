// spherical_harmonics_backward.cu

#include "checks.cuh"
#include "gsplat_cuda/cuda_backward.cuh"
#include "sphericart_cuda.hpp"
#include <thrust/device_vector.h>

__global__ void compute_sh_gradients_kernel(const float *d_sph, const float *rgb_grad_out, const int n_coeffs,
                                            const int N, float *sh_grad_in, float *sh_grad_band_0_in) {
  // Determine the unique index for this thread, corresponding to a single point/Gaussian.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  // Set up pointers to the data for the current point.
  const float *point_sph_vals = d_sph + idx * n_coeffs;
  const float *point_rgb_grad = rgb_grad_out + idx * 3;
  // Pointer for the new band 0 gradient output
  float *point_sh_grad_band_0 = sh_grad_band_0_in + idx * 3;

  // --- Gradient for Band 0 Coefficients ---
  // The gradient for the band 0 coefficient is simply the incoming gradient
  // from the logit, as its derivative in the forward pass is 1.
  point_sh_grad_band_0[0] = point_rgb_grad[0] * point_sph_vals[0];
  point_sh_grad_band_0[1] = point_rgb_grad[1] * point_sph_vals[0];
  point_sh_grad_band_0[2] = point_rgb_grad[2] * point_sph_vals[0];

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
    }
  }
}

void precompute_spherical_harmonics_backward(const float *const xyz_c, const float *const rgb_grad_out, const int l_max,
                                             const int N, float *sh_grad_in, float *sh_grad_band_0_in,
                                             cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(xyz_c);
  ASSERT_DEVICE_POINTER(rgb_grad_out);
  ASSERT_DEVICE_POINTER(sh_grad_band_0_in);
  ASSERT_DEVICE_POINTER(sh_grad_in);

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
      thrust::raw_pointer_cast(d_sph.data()), // Pass raw pointer
      rgb_grad_out, n_coeffs, N, sh_grad_in, sh_grad_band_0_in);
}
