// render_backward.cu

#include "checks.cuh"
#include "gsplat/cuda_backward.hpp"

constexpr int TILE_SIZE = 16;
constexpr int BATCH_SIZE = 960;

__device__ __forceinline__ float warpReduceSum(float val) {
  const unsigned int FULL_MASK = 0xffffffff;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }
  return val;
}

template <unsigned int CHUNK_SIZE>
__global__ void render_tiles_backward_kernel(
    const float *__restrict__ uvs, const float *__restrict__ opacity, const float *__restrict__ rgb,
    const float *__restrict__ conic, const float *__restrict__ view_dir_by_pixel,
    const int *__restrict__ splat_start_end_idx_by_tile_idx, const int *__restrict__ gaussian_idx_by_splat_idx,
    const float *__restrict__ background_rgb, const int *__restrict__ num_splats_per_pixel,
    const float *__restrict__ final_weight_per_pixel, const float *__restrict__ grad_image, const int image_width,
    const int image_height, float *__restrict__ grad_rgb, float *__restrict__ grad_opacity, float *__restrict__ grad_uv,
    float *__restrict__ grad_conic) {
  // Grid processes tiles, blocks process pixels within each tile
  const int u_splat = blockIdx.x * blockDim.x + threadIdx.x;
  const int v_splat = blockIdx.y * blockDim.y + threadIdx.y;
  const int linear_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  const unsigned int lane_id = linear_thread_id % 32;

  const int tile_idx = blockIdx.x + blockIdx.y * gridDim.x;
  const int splat_idx_start = splat_start_end_idx_by_tile_idx[tile_idx];
  const int splat_idx_end = splat_start_end_idx_by_tile_idx[tile_idx + 1];
  int num_splats_this_tile = splat_idx_end - splat_idx_start;

  // Per-pixel variables stored in registers
  bool background_initialized = false;
  int num_splats_this_pixel;
  float weight;
  float grad_image_local[3];
  float color_accum[3] = {0.0f, 0.0f, 0.0f};

  bool valid_pixel = u_splat < image_width && v_splat < image_height;

  if (valid_pixel) {
    num_splats_this_pixel = num_splats_per_pixel[v_splat * image_width + u_splat];
    weight = final_weight_per_pixel[u_splat + v_splat * image_width];
#pragma unroll
    for (int channel = 0; channel < 3; channel++) {
      grad_image_local[channel] = grad_image[(v_splat * image_width + u_splat) * 3 + channel];
    }
  }

  // Shared memory for batched processing of Gaussians
  __shared__ int _gaussian_idx_by_splat_idx[CHUNK_SIZE];
  __shared__ float _uvs[CHUNK_SIZE * 2];
  __shared__ float _opacity[CHUNK_SIZE];
  __shared__ float _rgb[CHUNK_SIZE * 3];
  __shared__ float _conic[CHUNK_SIZE * 3];

  const int num_chunks = (num_splats_this_tile + CHUNK_SIZE - 1) / CHUNK_SIZE;
  const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y;

  for (int chunk_idx = num_chunks - 1; chunk_idx >= 0; chunk_idx--) {
    __syncthreads();
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
    __syncthreads();

    int chunk_start = chunk_idx * CHUNK_SIZE;
    int chunk_end = min((chunk_idx + 1) * CHUNK_SIZE, num_splats_this_tile);
    for (int i = chunk_end - chunk_start - 1; i >= 0; i--) {
      const int tile_splat_idx = chunk_idx * CHUNK_SIZE + i;

      float grad_opa = 0.0f, grad_u = 0.0f, grad_v = 0.0f;
      float grad_rgb[3] = {0.0f, 0.0f, 0.0f};
      float grad_conic_splat[3] = {0.0f, 0.0f, 0.0f};

      if (valid_pixel && tile_splat_idx < num_splats_this_pixel) {
        const float u_mean = _uvs[i * 2 + 0];
        const float v_mean = _uvs[i * 2 + 1];
        const float u_diff = float(u_splat) - u_mean;
        const float v_diff = float(v_splat) - v_mean;

        float a = _conic[i * 3 + 0] + 0.25f;
        const float b = _conic[i * 3 + 1];
        float c = _conic[i * 3 + 2] + 0.25f;

        const float det = a * c - b * b;
        const float reciprocal_det = 1.0f / det;
        const float mh_sq = (c * u_diff * u_diff - 2.0f * b * u_diff * v_diff + a * v_diff * v_diff) * reciprocal_det;

        float norm_prob = 0.0f;
        if (mh_sq > 0.0f) {
          norm_prob = __expf(-0.5f * mh_sq);
        }

        float alpha = min(0.9999f, _opacity[i] * norm_prob);

        // No need for the lower alpha bound check with fast_exp
        const float reciprocal_one_minus_alpha = 1.0f / (1.0f - alpha);
        if (i < num_splats_this_pixel - 1) {
          weight *= reciprocal_one_minus_alpha;
        }

#pragma unroll
        for (int channel = 0; channel < 3; channel++) {
          grad_rgb[channel] = alpha * weight * grad_image_local[channel];
        }

        float grad_alpha = 0.0f;
#pragma unroll
        for (int channel = 0; channel < 3; channel++) {
          grad_alpha +=
              (_rgb[channel] * weight - color_accum[channel] * reciprocal_one_minus_alpha) * grad_image_local[channel];
        }

        grad_opa = norm_prob * grad_alpha;
        float grad_prob = _opacity[i] * grad_alpha;
        float grad_mh_sq = -0.5f * norm_prob * grad_prob;

        grad_u = -(2.0f * c * u_diff - 2.0f * b * v_diff) * reciprocal_det * grad_mh_sq;
        grad_v = -(2.0f * a * v_diff - 2.0f * b * u_diff) * reciprocal_det * grad_mh_sq;

        const float common_frac = -mh_sq * reciprocal_det * grad_mh_sq;
        grad_conic_splat[0] = (v_diff * v_diff * reciprocal_det - c * common_frac) * grad_mh_sq;
        grad_conic_splat[1] = (-2.0f * u_diff * v_diff * reciprocal_det + 2.0f * b * common_frac) * grad_mh_sq;
        grad_conic_splat[2] = (u_diff * u_diff * reciprocal_det - a * common_frac) * grad_mh_sq;

#pragma unroll
        for (int channel = 0; channel < 3; channel++) {
          color_accum[channel] += _rgb[channel] * alpha * weight;
        }
      }

      // --- Warp-Level Reduction ---
      grad_opa = warpReduceSum(grad_opa);
      grad_u = warpReduceSum(grad_u);
      grad_v = warpReduceSum(grad_v);

#pragma unroll
      for (int j = 0; j < 3; j++)
        grad_conic_splat[j] = warpReduceSum(grad_conic_splat[j]);

#pragma unroll
      for (int j = 0; j < 3; j++)
        grad_rgb[j] = warpReduceSum(grad_rgb[j]);

      // Lane 0 of the warp performs the atomic write
      if (lane_id == 0) {
        const int gaussian_idx = _gaussian_idx_by_splat_idx[i];
#pragma unroll
        for (int j = 0; j < 3; j++)
          atomicAdd(grad_rgb + gaussian_idx * 3 + j, grad_rgb[j]);
        atomicAdd(grad_opacity + gaussian_idx, grad_opa);
        atomicAdd(grad_uv + gaussian_idx * 2 + 0, grad_u);
        atomicAdd(grad_uv + gaussian_idx * 2 + 1, grad_v);
#pragma unroll
        for (int j = 0; j < 3; j++)
          atomicAdd(grad_conic + gaussian_idx * 3 + j, grad_conic_splat[j]);
      }
    }
  }
}

void render_image_backward(const float *const uvs, const float *const opacity, const float *const conic,
                           const float *const rgb, const float *const background_rgb,
                           const float *const view_dir_by_pixel, const int *const sorted_splats,
                           const int *const splat_range_by_tile, const int *const num_splats_per_pixel,
                           const float *const final_weight_per_pixel, const float *const grad_image,
                           const int image_width, const int image_height, const int l_max, float *grad_rgb,
                           float *grad_opacity, float *grad_uv, float *grad_conic, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(uvs);
  ASSERT_DEVICE_POINTER(opacity);
  ASSERT_DEVICE_POINTER(conic);
  ASSERT_DEVICE_POINTER(rgb);
  ASSERT_DEVICE_POINTER(background_rgb);
  ASSERT_DEVICE_POINTER(view_dir_by_pixel);
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
      uvs, opacity, rgb, conic, view_dir_by_pixel, splat_range_by_tile, sorted_splats, background_rgb,
      num_splats_per_pixel, final_weight_per_pixel, grad_image, image_width, image_height, grad_rgb, grad_opacity,
      grad_uv, grad_conic);
}
