// render_backward.cu

#include "checks.cuh"
#include "gsplat/cuda_backward.hpp"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

constexpr int TILE_SIZE = 16;
constexpr int BATCH_SIZE = 256;
constexpr int TG_SIZE = 32;

template <unsigned int CHUNK_SIZE>
__global__ void render_tiles_backward_kernel(
    const float *__restrict__ uvs, const float *__restrict__ opacity, const float *__restrict__ rgb,
    const float *__restrict__ conic, const int *__restrict__ splat_start_end_idx_by_tile_idx,
    const int *__restrict__ gaussian_idx_by_splat_idx, const float *__restrict__ background_rgb,
    const int *__restrict__ num_splats_per_pixel, const float *__restrict__ final_weight_per_pixel,
    const float *__restrict__ grad_image, const int image_width, const int image_height, float *__restrict__ grad_rgb,
    float *__restrict__ grad_opacity, float *__restrict__ grad_uv, float *__restrict__ grad_conic) {
  // Grid processes tiles, blocks process pixels within each tile
  const int u_splat = blockIdx.x * blockDim.x + threadIdx.x;
  const int v_splat = blockIdx.y * blockDim.y + threadIdx.y;
  const int pixel_id = v_splat * image_width + u_splat;

  const int tile_idx = blockIdx.x + blockIdx.y * gridDim.x;
  const int splat_idx_start = splat_start_end_idx_by_tile_idx[tile_idx];
  const int splat_idx_end = splat_start_end_idx_by_tile_idx[tile_idx + 1];
  int num_splats_this_tile = splat_idx_end - splat_idx_start;

  // Per-pixel variables stored in registers
  bool background_initialized = false;
  int num_splats_this_pixel;
  float T;
  float grad_image_local[3];
  float color_accum[3] = {0.0f, 0.0f, 0.0f};

  bool valid_pixel = u_splat < image_width && v_splat < image_height;

  if (valid_pixel) {
    num_splats_this_pixel = num_splats_per_pixel[pixel_id];
    T = final_weight_per_pixel[u_splat + v_splat * image_width];
#pragma unroll
    for (int channel = 0; channel < 3; channel++) {
      grad_image_local[channel] = grad_image[pixel_id * 3 + channel];
    }
  }

  // Shared memory for batched processing of Gaussians
  __shared__ int _gaussian_idx_by_splat_idx[CHUNK_SIZE];
  __shared__ float _uvs[CHUNK_SIZE * 2];
  __shared__ float _opacity[CHUNK_SIZE];
  __shared__ float _rgb[CHUNK_SIZE * 3];
  __shared__ float _conic[CHUNK_SIZE * 3];

  // Coopertive group at block level i.e tiles
  cg::thread_block tile_thread_group = cg::this_thread_block();

  const int num_chunks = (num_splats_this_tile + CHUNK_SIZE - 1) / CHUNK_SIZE;
  const int thread_id = tile_thread_group.thread_rank();
  const int block_size = blockDim.x * blockDim.y;

  // Iterate throught Gaussian chunks back to front
  for (int chunk_idx = num_chunks - 1; chunk_idx >= 0; chunk_idx--) {
    tile_thread_group.sync();
    // Load chunk to SMEM
    for (int i = thread_id; i < CHUNK_SIZE; i += block_size) {
      const int tile_splat_idx = chunk_idx * CHUNK_SIZE + i;
      if (tile_splat_idx >= num_splats_this_tile)
        break;
      const int global_splat_idx = splat_idx_start + tile_splat_idx;
      const int gaussian_idx = gaussian_idx_by_splat_idx[global_splat_idx];

      _gaussian_idx_by_splat_idx[i] = gaussian_idx;
      _uvs[i * 2 + 0] = uvs[gaussian_idx * 2 + 0];
      _uvs[i * 2 + 1] = uvs[gaussian_idx * 2 + 1];
      _opacity[i] = opacity[gaussian_idx];

#pragma unroll
      for (int j = 0; j < 3; j++)
        _rgb[i * 3 + j] = rgb[gaussian_idx * 3 + j];
#pragma unroll
      for (int j = 0; j < 3; j++)
        _conic[i * 3 + j] = conic[gaussian_idx * 3 + j];
    }
    tile_thread_group.sync();

    int chunk_start = chunk_idx * CHUNK_SIZE;
    int chunk_end = min((chunk_idx + 1) * CHUNK_SIZE, num_splats_this_tile);
    for (int i = chunk_end - chunk_start - 1; i >= 0; i--) {
      const int tile_splat_idx = chunk_idx * CHUNK_SIZE + i;

      float grad_opa = 0.0f, grad_u = 0.0f, grad_v = 0.0f;
      float grad_rgb_local[3] = {0.0f, 0.0f, 0.0f};
      float grad_conic_splat[3] = {0.0f, 0.0f, 0.0f};

      if (valid_pixel && tile_splat_idx < num_splats_this_pixel) {
        const float u_mean = _uvs[i * 2 + 0];
        const float v_mean = _uvs[i * 2 + 1];
        const float u_diff = float(u_splat) - u_mean;
        const float v_diff = float(v_splat) - v_mean;

        const float a = _conic[i * 3 + 0] + 0.25f;
        const float b = _conic[i * 3 + 1] + 0.5f;
        const float c = _conic[i * 3 + 2] + 0.25f;

        const float det = a * c - b * b;
        const float reciprocal_det = __frcp_rn(det);
        const float mh_sq = (c * u_diff * u_diff - 2.0f * b * u_diff * v_diff + a * v_diff * v_diff) * reciprocal_det;

        float norm_prob = 0.0f;
        if (mh_sq > 0.0f) {
          norm_prob = __expf(-0.5f * mh_sq);
        }

        float alpha = fminf(0.999f, _opacity[i] * norm_prob);
      }

      // --- Block-Level Reduction ---
      grad_opa = cg::reduce(tile_thread_group, grad_opa, cg::plus<float>());
      grad_u = cg::reduce(tile_thread_group, grad_u, cg::plus<float>());
      grad_v = cg::reduce(tile_thread_group, grad_v, cg::plus<float>());

#pragma unroll
      for (int j = 0; j < 3; j++)
        grad_conic_splat[j] = cg::reduce(tile_thread_group, grad_conic_splat[j], cg::plus<float>());

#pragma unroll
      for (int j = 0; j < 3; j++)
        grad_rgb_local[j] = cg::reduce(tile_thread_group, grad_rgb_local[j], cg::plus<float>());

      // Lane 0 of the warp performs the atomic write
      const int gaussian_idx = _gaussian_idx_by_splat_idx[i];
      if (thread_id == 0) {
        atomicAdd(grad_opacity + gaussian_idx, grad_opa);
        atomicAdd(grad_uv + gaussian_idx * 2 + 0, grad_u);
        atomicAdd(grad_uv + gaussian_idx * 2 + 1, grad_v);
#pragma unroll
        for (int j = 0; j < 3; j++)
          atomicAdd(grad_rgb + gaussian_idx * 3 + j, grad_rgb_local[j]);
#pragma unroll
        for (int j = 0; j < 3; j++)
          atomicAdd(grad_conic + gaussian_idx * 3 + j, grad_conic_splat[j]);
      }
    }
  }
}

void render_image_backward(const float *const uvs, const float *const opacity, const float *const conic,
                           const float *const rgb, const float *const background_rgb, const int *const sorted_splats,
                           const int *const splat_range_by_tile, const int *const num_splats_per_pixel,
                           const float *const final_weight_per_pixel, const float *const grad_image,
                           const int image_width, const int image_height, float *grad_rgb, float *grad_opacity,
                           float *grad_uv, float *grad_conic, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(uvs);
  ASSERT_DEVICE_POINTER(opacity);
  ASSERT_DEVICE_POINTER(conic);
  ASSERT_DEVICE_POINTER(rgb);
  ASSERT_DEVICE_POINTER(background_rgb);
  ASSERT_DEVICE_POINTER(sorted_splats);
  ASSERT_DEVICE_POINTER(splat_range_by_tile);
  ASSERT_DEVICE_POINTER(num_splats_per_pixel);
  ASSERT_DEVICE_POINTER(final_weight_per_pixel);
  ASSERT_DEVICE_POINTER(grad_image);
  ASSERT_DEVICE_POINTER(grad_rgb);
  ASSERT_DEVICE_POINTER(grad_opacity);
  ASSERT_DEVICE_POINTER(grad_uv);
  ASSERT_DEVICE_POINTER(grad_conic);

  const int num_tiles_x = (image_width + TILE_SIZE - 1) / TILE_SIZE;
  const int num_tiles_y = (image_height + TILE_SIZE - 1) / TILE_SIZE;

  dim3 block_size(TILE_SIZE, TILE_SIZE, 1);
  dim3 grid_size(num_tiles_x, num_tiles_y, 1);

  // Launch the single, non-templated kernel
  render_tiles_backward_kernel<BATCH_SIZE><<<grid_size, block_size, 0, stream>>>(
      uvs, opacity, rgb, conic, splat_range_by_tile, sorted_splats, background_rgb, num_splats_per_pixel,
      final_weight_per_pixel, grad_image, image_width, image_height, grad_rgb, grad_opacity, grad_uv, grad_conic);
}
