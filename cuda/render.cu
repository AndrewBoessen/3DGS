// render.cu

#include "checks.cuh"
#include "gsplat/cuda_forward.hpp"

constexpr int TILE_SIZE = 16;
constexpr int BATCH_SIZE = 960;

template <unsigned int splat_batch_size>
__global__ void
render_tiles_kernel(const float *__restrict__ uvs, const float *__restrict__ opacity, const float *__restrict__ rgb,
                    const float *__restrict__ conic, const int *__restrict__ splat_start_end_idx_by_tile_idx,
                    const int *__restrict__ gaussian_idx_by_splat_idx, const int image_width, const int image_height,
                    float *__restrict__ final_weight_per_pixel, float *__restrict__ image) {
  // grid = tiles, blocks = pixels within each tile
  const int u_splat = blockIdx.x * blockDim.x + threadIdx.x;
  const int v_splat = blockIdx.y * blockDim.y + threadIdx.y;
  const int tile_idx = blockIdx.x + blockIdx.y * gridDim.x;

  // keep threads around even if pixel is not valid for copying data
  bool valid_pixel = u_splat < image_width && v_splat < image_height;

  const int splat_idx_start = splat_start_end_idx_by_tile_idx[tile_idx];
  const int splat_idx_end = splat_start_end_idx_by_tile_idx[tile_idx + 1];
  int num_splats_this_tile = splat_idx_end - splat_idx_start;

  const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y;

  float alpha_accum = 0.0;
  float alpha_weight = 0.0;

  // shared memory copies of inputs
  __shared__ float _uvs[splat_batch_size * 2];
  __shared__ float _opacity[splat_batch_size];
  __shared__ float _rgb[splat_batch_size * 3];
  __shared__ float _conic[splat_batch_size * 3];

  const int shared_image_size = TILE_SIZE * TILE_SIZE * 3;
  __shared__ float _image[shared_image_size];

// init image values
#pragma unroll
  for (int i = thread_id; i < shared_image_size; i += block_size) {
    _image[i] = 0.0;
  }

  __syncthreads();
  const int num_batches = (num_splats_this_tile + splat_batch_size - 1) / splat_batch_size;
  for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    // copy data for current batch
    for (int i = thread_id; i < splat_batch_size; i += block_size) {
      const int tile_splat_idx = batch_idx * splat_batch_size + i;
      if (tile_splat_idx >= num_splats_this_tile) {
        break;
      }
      const int global_splat_idx = splat_idx_start + tile_splat_idx;

      const int gaussian_idx = gaussian_idx_by_splat_idx[global_splat_idx];
      _uvs[i * 2 + 0] = uvs[gaussian_idx * 2 + 0];
      _uvs[i * 2 + 1] = uvs[gaussian_idx * 2 + 1];
      _opacity[i] = opacity[gaussian_idx];

      for (int channel = 0; channel < 3; channel++) {
        _rgb[i * 3 + channel] = rgb[gaussian_idx * 3 + channel];
      }

#pragma unroll
      for (int j = 0; j < 3; j++) {
        _conic[i * 3 + j] = conic[gaussian_idx * 3 + j];
      }
    }
    __syncthreads(); // wait for copying to complete

    // mask invalid threads outside of image
    if (valid_pixel) {
      int batch_start = batch_idx * splat_batch_size;
      int batch_end = min((batch_idx + 1) * splat_batch_size, num_splats_this_tile);
      int num_splats_this_batch = batch_end - batch_start;

      for (int i = 0; i < num_splats_this_batch; i++) {
        if (alpha_accum > 0.9999) {
          // pixel value is already saturated
          break;
        }
        const float u_mean = _uvs[i * 2 + 0];
        const float v_mean = _uvs[i * 2 + 1];

        // distance from splat center
        const float u_diff = float(u_splat) - u_mean;
        const float v_diff = float(v_splat) - v_mean;

        // load conic values
        // add 0.25 for numeric stability with fast_exp
        const float a = _conic[i * 3 + 0] + 0.25;
        const float b = _conic[i * 3 + 1] + 0.5;
        const float c = _conic[i * 3 + 2] + 0.25;

        const float det = a * c - b * b;
        // compute mahalanobis distance
        const float mh_sq = (c * u_diff * u_diff - (b + b) * u_diff * v_diff + a * v_diff * v_diff) / det;

        float alpha = 0.0;
        // normalize
        if (mh_sq > 0.0) {
          alpha = _opacity[i] * __expf(-0.5 * mh_sq);
        }
        alpha_weight = 1.0 - alpha_accum;
        const float weight = alpha * (1.0 - alpha_accum);

        // update rgb values
        const int base_image_id = threadIdx.y * TILE_SIZE + threadIdx.x;
        const int base_rgb_id = i * 3;
        _image[base_image_id + 0] += _rgb[base_rgb_id + 0] * weight; // R
        _image[base_image_id + 1] += _rgb[base_rgb_id + 1] * weight; // G
        _image[base_image_id + 2] += _rgb[base_rgb_id + 2] * weight; // B

        // update alpha_accum
        alpha_accum += weight;
      }
      // wait for all threads to complete current batch
      __syncthreads();
    }
  }
  // add background color at end
  const int base_image_id = (threadIdx.y * TILE_SIZE + threadIdx.x) * 3;
  if (valid_pixel) {
    _image[base_image_id + 0] += (1.0 - alpha_accum); // R
    _image[base_image_id + 1] += (1.0 - alpha_accum); // G
    _image[base_image_id + 2] += (1.0 - alpha_accum); // B
  }

  __syncthreads();
  // copy back to global memory
  if (valid_pixel) {
    final_weight_per_pixel[v_splat * image_width + u_splat] = alpha_weight;

    image[(v_splat * image_width + u_splat) * 3 + 0] = _image[base_image_id * 3 + 0]; // R
    image[(v_splat * image_width + u_splat) * 3 + 1] = _image[base_image_id * 3 + 1]; // G
    image[(v_splat * image_width + u_splat) * 3 + 2] = _image[base_image_id * 3 + 2]; // B
  }
}

void render_image(const float *uv, const float *opacity, const float *conic, const float *rgb, const int *sorted_splats,
                  const int *splat_range_by_tile, const int image_width, const int image_height,
                  float *weight_per_pixel, float *image) {
  ASSERT_DEVICE_POINTER(uv);
  ASSERT_DEVICE_POINTER(opacity);
  ASSERT_DEVICE_POINTER(conic);
  ASSERT_DEVICE_POINTER(rgb);
  ASSERT_DEVICE_POINTER(sorted_splats);
  ASSERT_DEVICE_POINTER(splat_range_by_tile);
  ASSERT_DEVICE_POINTER(weight_per_pixel);
  ASSERT_DEVICE_POINTER(image);

  int num_tiles_x = (image_width + TILE_SIZE - 1) / TILE_SIZE;
  int num_tiles_y = (image_height + TILE_SIZE - 1) / TILE_SIZE;

  dim3 block_size(TILE_SIZE, TILE_SIZE, 1);
  dim3 grid_size(num_tiles_x, num_tiles_y, 1);

  render_tiles_kernel<BATCH_SIZE><<<grid_size, block_size>>>(
      uv, opacity, rgb, conic, splat_range_by_tile, sorted_splats, image_width, image_height, weight_per_pixel, image);
}
