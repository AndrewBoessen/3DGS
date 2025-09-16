// spherical_harmonics_backward.cu

#include "checks.cuh"
#include "gsplat/cuda_backward.hpp"
#include "sphericart_cuda.hpp"

__global__ void compute_sh_gradients_kernel(const float *d_sph, const float *rgb_grad_out, const int n_coeffs,
                                            const int N, float *sh_grad_in) {
  // Determine the unique index for this thread, corresponding to a single point/Gaussian.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  // Set up pointers to the data for the current point.
  const float *point_sph_vals = d_sph + idx * n_coeffs;
  const float *point_rgb_grad = rgb_grad_out + idx * 3;
  float *point_sh_grad = sh_grad_in + idx * n_coeffs * 3;

  // For each SH coefficient, calculate its gradient.
  // The chain rule simplifies here: dL/d(coeff) = dL/d(logit) * d(logit)/d(coeff).
  // Since logit = sum(coeff * sh_val), d(logit)/d(coeff) is simply sh_val.
  for (int i = 0; i < n_coeffs; ++i) {
    float sh_val = point_sph_vals[i];
    point_sh_grad[i * 3 + 0] = point_rgb_grad[0] * sh_val; // Gradient for Red channel's coefficient
    point_sh_grad[i * 3 + 1] = point_rgb_grad[1] * sh_val; // Gradient for Green channel's coefficient
    point_sh_grad[i * 3 + 2] = point_rgb_grad[2] * sh_val; // Gradient for Blue channel's coefficient
  }
}

void precompute_spherical_harmonics_backward(const float *const xyz_c, const float *const rgb_grad_out, const int l_max,
                                             const int N, float *sh_grad_in, cudaStream_t stream) {
  // Initialize the sphericart calculator for the given maximum degree.
  sphericart::cuda::SphericalHarmonics<float> calculator_cuda(l_max);

  const int n_coeffs = (l_max + 1) * (l_max + 1);

  // Allocate temporary device memory for SH basis values (d_sph) and their derivatives
  // with respect to Cartesian coordinates (d_dsph).
  float *d_sph;
  float *d_dsph;
  CHECK_CUDA(cudaMalloc(&d_sph, N * n_coeffs * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_dsph, N * n_coeffs * 3 * sizeof(float)));

  // Use the sphericart library to compute the SH basis values.
  // Although this function also computes gradients with respect to xyz (d_dsph),
  // they are not needed for this particular backward pass but are computed as requested.
  calculator_cuda.compute_with_gradients(xyz_c, N, d_sph, d_dsph, stream);

  // Define CUDA kernel launch parameters.
  const int blockSize = 256;
  const int gridSize = (N + blockSize - 1) / blockSize;

  // Launch the kernel to compute the final SH coefficient gradients.
  compute_sh_gradients_kernel<<<gridSize, blockSize, 0, stream>>>(d_sph, rgb_grad_out, n_coeffs, N, sh_grad_in);

  // Free the temporary device memory.
  CHECK_CUDA(cudaFree(d_sph));
  CHECK_CUDA(cudaFree(d_dsph));
}
