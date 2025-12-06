// render.cu

#include "checks.cuh"
#include "gsplat_cuda/cuda_forward.cuh"

__global__ void render_tiles_kernel(const int num_tiles_x, const int num_tiles_y, const float *__restrict__ uvs,
                                    const float *__restrict__ opacity, const float *__restrict__ rgb,
                                    const float *__restrict__ conic, const float background_opacity,
                                    const int *__restrict__ splat_start_end_idx_by_tile_idx,
                                    const int *__restrict__ gaussian_idx_by_splat_idx, const int image_width,
                                    const int image_height, int *__restrict__ splats_per_pixel,
                                    float *__restrict__ final_weight_per_pixel, float *__restrict__ image) {
  // grid = tiles, blocks = pixels within each tile
  const int PIXELS_PER_THREAD = (TILE_SIZE_FWD * TILE_SIZE_FWD) / 32;
  const int tile_idx = blockIdx.x * blockDim.y + threadIdx.y;

  // Tile outside of image
  if (tile_idx >= num_tiles_x * num_tiles_y)
    return;

  const int in_tile_x = threadIdx.x % TILE_SIZE_FWD;                     // local tile x
  const int in_tile_y = threadIdx.x / TILE_SIZE_FWD * PIXELS_PER_THREAD; // local tile y

  const int base_pixel_x = (tile_idx % num_tiles_x) * TILE_SIZE_FWD + in_tile_x;
  const int base_pixel_y = (tile_idx / num_tiles_x) * TILE_SIZE_FWD + in_tile_y;

  const int splat_idx_start = splat_start_end_idx_by_tile_idx[tile_idx];
  const int splat_idx_end = splat_start_end_idx_by_tile_idx[tile_idx + 1];
  const int total_splats = splat_idx_end - splat_idx_start;

  // Pixel-local accumulators
  float T[PIXELS_PER_THREAD];
  float3 accumulated_rgb[PIXELS_PER_THREAD];
  int num_splats[PIXELS_PER_THREAD];

#pragma unroll
  for (int i = 0; i < PIXELS_PER_THREAD; i++) {
    T[i] = 1.0f;
    accumulated_rgb[i] = {0.0f, 0.0f, 0.0f};
    num_splats[i] = 0;
  }

  unsigned int any_active = 0xFFFFFFFF;
  int index_in_tile = 0;
  const int *splats_in_tile = &gaussian_idx_by_splat_idx[splat_idx_start];
  bool done = false;

  // Iterate on splats in the tile front to back
  for (; (index_in_tile < total_splats) && (any_active != 0); index_in_tile++) {
    const int gaussian_idx = splats_in_tile[index_in_tile];
    float basic;
    float linear;
    float quad;

    float3 color = {rgb[gaussian_idx * 3 + 0], rgb[gaussian_idx * 3 + 1], rgb[gaussian_idx * 3 + 2]};
    float opa = 1.0f / (1.0f + __expf(-opacity[gaussian_idx]));
    float2 d = {0.0f, 0.0f};
    d.x = uvs[gaussian_idx * 2 + 0] - (float)base_pixel_x;
    d.y = uvs[gaussian_idx * 2 + 1] - (float)base_pixel_y;

    const float inv_cov00 = conic[gaussian_idx * 3 + 0];
    const float inv_cov01 = conic[gaussian_idx * 3 + 1];
    const float inv_cov11 = conic[gaussian_idx * 3 + 2];
    basic = -0.5f * (inv_cov00 * d.x * d.x + 2.0f * inv_cov01 * d.x * d.y + inv_cov11 * d.y * d.y);
    linear = inv_cov11 * d.y + inv_cov01 * d.x;
    quad = -0.5f * inv_cov11;

    any_active = 0;
    for (int i = 0; i < PIXELS_PER_THREAD; i++) {
      const float power = fminf(0.0f, basic + linear * i + quad * i * i);

      float alpha = fminf(0.99f, opa * __expf(power));
      alpha = (alpha > 0.00392156862f) ? !done * alpha : 0.0f;

      const float test_T = T[i] * (1.0f - alpha);
      done = test_T < 0.0001f;

      any_active |= __ballot_sync(0xFFFFFFFF, !done);

      const float weight = alpha * T[i];

      accumulated_rgb[i].x += color.x * weight;
      accumulated_rgb[i].y += color.y * weight;
      accumulated_rgb[i].z += color.z * weight;

      T[i] = test_T;
      num_splats[i] += !done;
    }
  }

  // Write results to output image
  for (int i = 0; i < PIXELS_PER_THREAD; i++) {
    const int global_pixel_x = base_pixel_x;
    const int global_pixel_y = base_pixel_y + i;

    const bool valid_pixel = global_pixel_x < image_width && global_pixel_y < image_height;

    if (valid_pixel) {
      splats_per_pixel[global_pixel_y * image_width + global_pixel_x] = num_splats[i];
      final_weight_per_pixel[global_pixel_y * image_width + global_pixel_x] = T[i];

      const int pixel_idx = (global_pixel_y * image_width + global_pixel_x) * 3;
      image[pixel_idx + 0] = accumulated_rgb[i].x + T[i] * background_opacity; // R
      image[pixel_idx + 1] = accumulated_rgb[i].y + T[i] * background_opacity; // G
      image[pixel_idx + 2] = accumulated_rgb[i].z + T[i] * background_opacity; // B
    }
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

  const int tiles_per_block = 4;
  const int num_tiles_x = (image_width + TILE_SIZE_FWD - 1) / TILE_SIZE_FWD;
  const int num_tiles_y = (image_height + TILE_SIZE_FWD - 1) / TILE_SIZE_FWD;
  const int total_blocks = (num_tiles_x * num_tiles_y + tiles_per_block - 1) / tiles_per_block;

  dim3 block_size(32, tiles_per_block, 1);
  dim3 grid_size(total_blocks, 1, 1);

  render_tiles_kernel<<<grid_size, block_size, 0, stream>>>(
      num_tiles_x, num_tiles_y, uv, opacity, rgb, conic, background_opacity, splat_range_by_tile, sorted_splats,
      image_width, image_height, splats_per_pixel, weight_per_pixel, image);
}
