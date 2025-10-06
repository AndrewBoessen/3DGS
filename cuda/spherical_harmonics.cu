// spherical_harmonics.cu

#include "checks.cuh"
#include "gsplat/cuda_forward.hpp"
#include "sphericart_cuda.hpp"

__global__ void compute_rgb_from_sh_kernel(const float *sh_coefficients, const float *sh_coeffs_band_0,
                                           const float *d_sph, const int n_coeffs, const int N, float *rgb) {
  // Determine the unique index for this thread
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  // Pointers to the start of the data for this specific point
  const float *point_sh_coeffs = sh_coefficients + idx * (n_coeffs - 1) * 3; // subtract 1 to account for zero band
  const float *point_sph_vals = d_sph + idx * n_coeffs;
  const float *point_sh_coeffs_band_0 = sh_coeffs_band_0 + idx * 3;

  // Initialize sums for each color channel with the band 0 coefficients
  float r = point_sh_coeffs_band_0[0] * point_sph_vals[0];
  float g = point_sh_coeffs_band_0[1] * point_sph_vals[0];
  float b = point_sh_coeffs_band_0[2] * point_sph_vals[0];

  // Sum the contributions from higher-order SH coefficients (l > 0)
  for (int i = 0; i < n_coeffs - 1; ++i) { // Start loop from 1
    float sh_val = point_sph_vals[i + 1];
    r += point_sh_coeffs[i * 3 + 0] * sh_val;
    g += point_sh_coeffs[i * 3 + 1] * sh_val;
    b += point_sh_coeffs[i * 3 + 2] * sh_val;
  }

  // Apply sigmoid activation to map the output to the [0, 1] range
  rgb[idx * 3 + 0] = __frcp_rn(1.0f + expf(-r));
  rgb[idx * 3 + 1] = __frcp_rn(1.0f + expf(-g));
  rgb[idx * 3 + 2] = __frcp_rn(1.0f + expf(-b));
}

void precompute_spherical_harmonics(const float *xyz, const float *sh_coefficients, const float *sh_coeffs_band_0,
                                    const int l_max, const int N, float *rgb, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(xyz);
  ASSERT_DEVICE_POINTER(sh_coefficients);
  // Assert that the new parameter is a valid device pointer
  ASSERT_DEVICE_POINTER(sh_coeffs_band_0);
  ASSERT_DEVICE_POINTER(rgb);

  // internal buffers and numerical factors are initalized at construction
  sphericart::cuda::SphericalHarmonics<float> calculator_cuda(l_max);

  const int n_coeffs = (l_max + 1) * (l_max + 1);

  float *d_sph;
  CHECK_CUDA(cudaMalloc(&d_sph, N * n_coeffs * sizeof(float)));

  // compute SH values
  calculator_cuda.compute(xyz, N, d_sph);

  // Define CUDA kernel launch parameters
  const int blockSize = 256;
  const int gridSize = (N + blockSize - 1) / blockSize;

  // Launch the kernel to compute the final RGB values
  compute_rgb_from_sh_kernel<<<gridSize, blockSize, 0, stream>>>(sh_coefficients, sh_coeffs_band_0, d_sph, n_coeffs, N,
                                                                 rgb);

  // Free the temporary device memory used for SH basis functions
  CHECK_CUDA(cudaFree(d_sph));
}
