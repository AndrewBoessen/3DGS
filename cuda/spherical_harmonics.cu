// spherical_harmonics.cu

#include "checks.cuh"
#include "gsplat/cuda_forward.hpp"
#include "sphericart_cuda.hpp"

__global__ void compute_rgb_from_sh_kernel(const float *sh_coefficients, const float *d_sph, const int n_coeffs,
                                           const int N, float *rgb) {
  // Determine the unique index for this thread
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  // Pointers to the start of the data for this specific point
  const float *point_sh_coeffs = sh_coefficients + idx * n_coeffs * 3;
  const float *point_sph_vals = d_sph + idx * n_coeffs;

  // Initialize sums for each color channel
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;

  // Sum the contributions from all SH coefficients
  for (int i = 0; i < n_coeffs; ++i) {
    float sh_val = point_sph_vals[i];
    r += point_sh_coeffs[i * 3 + 0] * sh_val;
    g += point_sh_coeffs[i * 3 + 1] * sh_val;
    b += point_sh_coeffs[i * 3 + 2] * sh_val;
  }

  // Apply sigmoid activation to map the output to the [0, 1] range
  const float one = 1.0f;
  rgb[idx * 3 + 0] = one / (one + expf(-r));
  rgb[idx * 3 + 1] = one / (one + expf(-g));
  rgb[idx * 3 + 2] = one / (one + expf(-b));
}

void precompute_spherical_harmonics(const float *xyz, const float *sh_coefficients, const int l_max, const int N,
                                    float *rgb, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(xyz);
  ASSERT_DEVICE_POINTER(sh_coefficients);
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
  compute_rgb_from_sh_kernel<<<gridSize, blockSize, 0, stream>>>(sh_coefficients, d_sph, n_coeffs, N, rgb);

  // Free the temporary device memory used for SH basis functions
  CHECK_CUDA(cudaFree(d_sph));
}
