// render.cu

#include "checks.cuh"
#include "gsplat/cuda_forward.hpp"

constexpr int BATCH_SIZE = 256;

template <unsigned int splat_batch_size>
__global__ void render_tiles_kernel(const float *__restrict__ uvs, const float *__restrict__ opacity,
                                    const float *__restrict__ rgb, const float *__restrict__ conic,
                                    const float background_opacity,
                                    const int *__restrict__ splat_start_end_idx_by_tile_idx,
                                    const int *__restrict__ gaussian_idx_by_splat_idx, const int image_width,
                                    const int image_height, int *__restrict__ splats_per_pixel,
                                    float *__restrict__ final_weight_per_pixel, float *__restrict__ image) {
  // grid = tiles, blocks = pixels within each tile
  const int u_splat = blockIdx.x * blockDim.x + threadIdx.x;
  const int v_splat = blockIdx.y * blockDim.y + threadIdx.y;
  const int tile_idx = blockIdx.x + blockIdx.y * gridDim.x;

  // Mask threads outside the image boundary
  const bool valid_pixel = u_splat < image_width && v_splat < image_height;

  const int splat_idx_start = splat_start_end_idx_by_tile_idx[tile_idx];
  const int splat_idx_end = splat_start_end_idx_by_tile_idx[tile_idx + 1];
  const int num_splats_this_tile = splat_idx_end - splat_idx_start;

  const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y;

  // Pixel-local accumulators
  float alpha_accum = 0.0f;
  float3 accumulated_rgb = {0.0f, 0.0f, 0.0f};
  int num_splats = 0;

  // shared memory copies of inputs
  __shared__ float _uvs[splat_batch_size * 2];
  __shared__ float _opacity[splat_batch_size];
  __shared__ float _rgb[splat_batch_size * 3];
  __shared__ float _conic[splat_batch_size * 3];

  const int num_batches = (num_splats_this_tile + splat_batch_size - 1) / splat_batch_size;
  for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    // Cooperatively load a batch of splat data into shared memory
    for (int i = thread_id; i < splat_batch_size; i += block_size) {
      const int tile_splat_idx = batch_idx * splat_batch_size + i;
      if (tile_splat_idx < num_splats_this_tile) {
        const int global_splat_idx = splat_idx_start + tile_splat_idx;

        const int gaussian_idx = gaussian_idx_by_splat_idx[global_splat_idx];
        _uvs[i * 2 + 0] = uvs[gaussian_idx * 2 + 0];
        _uvs[i * 2 + 1] = uvs[gaussian_idx * 2 + 1];
        _opacity[i] = opacity[gaussian_idx];

#pragma unroll
        for (int channel = 0; channel < 3; channel++) {
          _rgb[i * 3 + channel] = rgb[gaussian_idx * 3 + channel];
        }

#pragma unroll
        for (int j = 0; j < 3; j++) {
          _conic[i * 3 + j] = conic[gaussian_idx * 3 + j];
        }
      }
    }
    __syncthreads();

    // Mask invalid threads outside of image
    if (valid_pixel) {
      const int num_splats_this_batch = min(splat_batch_size, num_splats_this_tile - batch_idx * splat_batch_size);

      for (int i = 0; i < num_splats_this_batch; i++) {
        // Early exit if pixel is saturated
        if (alpha_accum > 0.999f) {
          break;
        }
        const float u_mean = _uvs[i * 2 + 0];
        const float v_mean = _uvs[i * 2 + 1];

        // Distance from splat center
        const float u_diff = float(u_splat) - u_mean;
        const float v_diff = float(v_splat) - v_mean;

        // Load conic values
        const float a = _conic[i * 3 + 0];
        const float b = _conic[i * 3 + 1];
        const float c = _conic[i * 3 + 2];

        const float det = a * c - b * b;
        if (det <= 0.0f)
          continue; // Skip degenerate or invalid Gaussians
        const float inv_det = __frcp_rn(det);

        // Compute Mahalanobis distance squared: d^2 = (x-μ)^T Σ^-1 (x-μ)
        const float mh_sq = inv_det * (c * u_diff * u_diff - 2.0f * b * u_diff * v_diff + a * v_diff * v_diff);
        if (mh_sq <= 0.0f) {
          num_splats++;
          continue; // Gaussian has no influence
        }

        // Apply sigmoid to opacity
        float opa = __frcp_rn(1.0f + __expf(-_opacity[i]));
        // Calculate alpha based on opacity and Gaussian falloff
        const float alpha = opa * __expf(-0.5f * mh_sq);

        // Alpha blending: C_out = α * C_in + (1 - α) * C_bg
        const float weight = alpha * (1.0f - alpha_accum);

        const int base_rgb_id = i * 3;
        accumulated_rgb.x += _rgb[base_rgb_id + 0] * weight;
        accumulated_rgb.y += _rgb[base_rgb_id + 1] * weight;
        accumulated_rgb.z += _rgb[base_rgb_id + 2] * weight;

        alpha_accum += weight;
        num_splats++;
      }
    }
    __syncthreads(); // Ensure all threads finish before loading the next batch
  }

  // Final write to global memory
  if (valid_pixel) {
    splats_per_pixel[v_splat * image_width + u_splat] = num_splats;
    final_weight_per_pixel[v_splat * image_width + u_splat] = 1.0f - alpha_accum;

    const float background_contribution = 1.0f - alpha_accum;
    const int pixel_idx = (v_splat * image_width + u_splat) * 3;
    image[pixel_idx + 0] = accumulated_rgb.x + background_opacity * background_contribution; // R
    image[pixel_idx + 1] = accumulated_rgb.y + background_opacity * background_contribution; // G
    image[pixel_idx + 2] = accumulated_rgb.z + background_opacity * background_contribution; // B
  }
}

void render_image(const float *uv, const float *opacity, const float *conic, const float *rgb,
                  const float background_opacity, const int *sorted_splats, const int *splat_range_by_tile,
                  const int image_width, const int image_height, int *splats_per_pixel, float *weight_per_pixel,
                  float *image, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(uv);
  ASSERT_DEVICE_POINTER(opacity);
  ASSERT_DEVICE_POINTER(conic);
  ASSERT_DEVICE_POINTER(rgb);
  ASSERT_DEVICE_POINTER(sorted_splats);
  ASSERT_DEVICE_POINTER(splat_range_by_tile);
  ASSERT_DEVICE_POINTER(splats_per_pixel);
  ASSERT_DEVICE_POINTER(weight_per_pixel);
  ASSERT_DEVICE_POINTER(image);

  int num_tiles_x = (image_width + TILE_SIZE_FWD - 1) / TILE_SIZE_FWD;
  int num_tiles_y = (image_height + TILE_SIZE_FWD - 1) / TILE_SIZE_FWD;

  dim3 block_size(TILE_SIZE_FWD, TILE_SIZE_FWD, 1);
  dim3 grid_size(num_tiles_x, num_tiles_y, 1);

  render_tiles_kernel<BATCH_SIZE><<<grid_size, block_size, 0, stream>>>(
      uv, opacity, rgb, conic, background_opacity, splat_range_by_tile, sorted_splats, image_width, image_height,
      splats_per_pixel, weight_per_pixel, image);
}
