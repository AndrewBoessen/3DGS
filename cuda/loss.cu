// loss.cu

#include "checks.cuh"
#include "gsplat/cuda_forward.cuh"
#include "thrust/detail/raw_pointer_cast.h"
#include <thrust/device_vector.h>

// Define constants for SSIM calculation.
namespace SSIMConstants {
constexpr int WINDOW_SIZE = 11;
constexpr int WINDOW_RADIUS = WINDOW_SIZE / 2;
constexpr float K1 = 0.01f;
constexpr float K2 = 0.03f;
// Dynamic range of pixel values (assuming normalized to [0,1]).
constexpr float L = 1.0f;
constexpr float C1 = (K1 * L) * (K1 * L);
constexpr float C2 = (K2 * L) * (K2 * L);
} // namespace SSIMConstants

__device__ float gaussian(int x, int y, float sigma) {
  float coeff = 1.0f / (2.0f * M_PI * sigma * sigma);
  float exponent = -(x * x + y * y) / (2.0f * sigma * sigma);
  return coeff * expf(exponent);
}

__device__ float l1_loss(const float pred, const float gt) { return fabsf(pred - gt); }

__global__ void fused_loss_kernel(const float *__restrict__ image, const float *__restrict__ gt_image, const int rows,
                                  const int cols, const float ssim_weight, float *__restrict__ image_grad,
                                  float *__restrict__ total_loss_ptr) {
  // --- Shared memory for block-level loss reduction ---
  extern __shared__ float s_loss[];
  const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
  s_loss[tid] = 0.0f;

  // --- Global thread indexing ---
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows) {
    return;
  }
  const int idx = y * cols + x;
  const float grad_scale = 1.0f / (float)(rows * cols);

  float pixel_l1_loss = 0.0f;
  float pixel_ssim_loss = 0.0f;

  // --- L1 Loss and Gradient Calculation ---
  for (int c = 0; c < 3; ++c) {
    int channel_idx = c * rows * cols + idx;
    float pred = image[channel_idx];
    float gt = gt_image[channel_idx];

    pixel_l1_loss += l1_loss(pred, gt);

    // Initialize gradient with the weighted L1 part. Gradient of L1 is sign(pred - gt).
    if (ssim_weight < 1.0f) {
      float grad_l1 = (pred > gt) ? 1.0f : -1.0f;
      if (pred == gt)
        grad_l1 = 0.0f;
      image_grad[channel_idx] = (1.0f - ssim_weight) * grad_l1 * grad_scale;
    } else {
      image_grad[channel_idx] = 0.0f;
    }
  }
  pixel_l1_loss /= 3.0f; // Average L1 loss over channels

  // --- SSIM Loss and Gradient Calculation (if weighted) ---
  if (ssim_weight > 0.0f) {
    float avg_ssim = 0.0f;

    // Process each channel independently for SSIM
    for (int c = 0; c < 3; ++c) {
      int channel_offset = c * rows * cols;
      const float gauss_sigma = 1.5f;

      // --- Window statistics calculation (Single Pass) ---
      float mu_p = 0.0f, mu_g = 0.0f;
      float p_sq_sum = 0.0f, g_sq_sum = 0.0f, pg_sum = 0.0f;
      float total_weight = 0.0f;

      for (int j = -SSIMConstants::WINDOW_RADIUS; j <= SSIMConstants::WINDOW_RADIUS; ++j) {
        for (int i = -SSIMConstants::WINDOW_RADIUS; i <= SSIMConstants::WINDOW_RADIUS; ++i) {
          int cur_x = x + i;
          int cur_y = y + j;

          // Clamp coordinates to image boundaries
          cur_x = max(0, min(cols - 1, cur_x));
          cur_y = max(0, min(rows - 1, cur_y));

          int window_idx = channel_offset + cur_y * cols + cur_x;
          float weight = gaussian(i, j, gauss_sigma);

          float p = image[window_idx];
          float g = gt_image[window_idx];

          mu_p += p * weight;
          mu_g += g * weight;
          p_sq_sum += p * p * weight;
          g_sq_sum += g * g * weight;
          pg_sum += p * g * weight;
          total_weight += weight;
        }
      }

      mu_p /= total_weight;
      mu_g /= total_weight;
      float var_p = p_sq_sum / total_weight - mu_p * mu_p;
      float var_g = g_sq_sum / total_weight - mu_g * mu_g;
      float cov_pg = pg_sum / total_weight - mu_p * mu_g;

      // --- SSIM Calculation ---
      float ssim_num = (2.0f * mu_p * mu_g + SSIMConstants::C1) * (2.0f * cov_pg + SSIMConstants::C2);
      float ssim_den = (mu_p * mu_p + mu_g * mu_g + SSIMConstants::C1) * (var_p + var_g + SSIMConstants::C2);
      float ssim = ssim_num / ssim_den;
      avg_ssim += ssim;

      // --- SSIM Gradient Calculation ---
      // Gradient of L_ssim = 1 - SSIM is d(1-SSIM)/dp_i = -d(SSIM)/dp_i
      // Using quotient rule: d(N/D)/dp_i = (D*dN/dp_i - N*dD/dp_i) / D^2
      int central_idx = channel_offset + idx;
      float p_i = image[central_idx];
      float g_i = gt_image[central_idx];
      float w_i = gaussian(0, 0, gauss_sigma) / total_weight;

      // Derivatives of stats w.r.t central predicted pixel p_i
      float d_mu_p = w_i;
      float d_var_p = 2.0f * (p_i - mu_p) * w_i;
      float d_cov_pg = (g_i - mu_g) * w_i;

      // Derivatives of numerator (N) and denominator (D) terms of SSIM
      float dN = (2.0f * mu_g * d_mu_p) * (2.0f * cov_pg + SSIMConstants::C2) +
                 (2.0f * mu_p * mu_g + SSIMConstants::C1) * (2.0f * d_cov_pg);
      float dD = (2.0f * mu_p * d_mu_p) * (var_p + var_g + SSIMConstants::C2) +
                 (mu_p * mu_p + mu_g * mu_g + SSIMConstants::C1) * d_var_p;

      float d_ssim = (dN * ssim_den - ssim_num * dD) / (ssim_den * ssim_den);

      // Add weighted SSIM gradient part. Grad of loss (1-SSIM) is -d_ssim.
      image_grad[central_idx] -= ssim_weight * 0.5f * d_ssim * grad_scale;
    }
    avg_ssim /= 3.0f;
    pixel_ssim_loss = (1.0f - avg_ssim) / 2.0f;
  }

  // --- Combine losses and store in shared memory for reduction ---
  float combined_loss = (1.0f - ssim_weight) * pixel_l1_loss + ssim_weight * pixel_ssim_loss;
  s_loss[tid] = combined_loss;

  __syncthreads();

  // --- Block-level reduction using shared memory ---
  for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_loss[tid] += s_loss[tid + s];
    }
    __syncthreads();
  }

  // First thread in block writes block's sum to global memory atomically
  if (tid == 0) {
    atomicAdd(total_loss_ptr, s_loss[0]);
  }
}

float fused_loss(const float *predicted_data, const float *gt_data, int rows, int cols, int channels,
                 const float ssim_weight, float *image_grad, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(predicted_data);
  ASSERT_DEVICE_POINTER(gt_data);
  ASSERT_DEVICE_POINTER(image_grad);

  dim3 blockDim(16, 16);
  dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

  // --- Allocate memory for total loss on device ---
  thrust::device_vector<float> d_total_loss(1, 0.0f);

  size_t shared_mem_size = blockDim.x * blockDim.y * sizeof(float);

  fused_loss_kernel<<<gridDim, blockDim, shared_mem_size, stream>>>(
      predicted_data, gt_data, rows, cols, ssim_weight, image_grad, thrust::raw_pointer_cast(d_total_loss.data()));

  // --- Copy result back to host ---
  float h_total_loss = d_total_loss[0];

  // Return the mean loss per pixel
  return h_total_loss / (rows * cols);
}
