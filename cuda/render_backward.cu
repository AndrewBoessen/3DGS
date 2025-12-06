// render_backward.cu

#include "checks.cuh"
#include "gsplat_cuda/cuda_backward.cuh"
#include <cassert>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__global__ void render_tiles_backward_kernel(
    const int num_tiles_x, const int num_tiles_y, const float *__restrict__ uvs, const float *__restrict__ opacity,
    const float *__restrict__ rgb, const float *__restrict__ conic,
    const int *__restrict__ splat_start_end_idx_by_tile_idx, const int *__restrict__ gaussian_idx_by_splat_idx,
    const float background_opacity, const int *__restrict__ num_splats_per_pixel,
    const float *__restrict__ final_weight_per_pixel, const float *__restrict__ grad_image, const int image_width,
    const int image_height, float *__restrict__ grad_rgb, float *__restrict__ grad_opacity, float *__restrict__ grad_uv,
    float *__restrict__ grad_conic) {
  // Grid processes tiles, blocks process pixels within each tile
  const int PIXELS_PER_THREAD = (TILE_SIZE_BWD * TILE_SIZE_BWD) / 32;
  const int tile_idx = blockIdx.x * blockDim.y + threadIdx.y;

  auto tile_thread_group = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(tile_thread_group);

  // Tile outside of image
  if (tile_idx >= num_tiles_x * num_tiles_y)
    return;

  const int splat_idx_start = splat_start_end_idx_by_tile_idx[tile_idx];
  const int splat_idx_end = splat_start_end_idx_by_tile_idx[tile_idx + 1];

  __shared__ float _image_grad[3][PIXELS_PER_THREAD][4 * 32];
  __shared__ float _trans_final[PIXELS_PER_THREAD][4 * 32];
  __shared__ unsigned int _splats_per_pixel[PIXELS_PER_THREAD][4 * 32];

  // Per-pixel variables stored in registers
  float T[PIXELS_PER_THREAD];
  float T_final[PIXELS_PER_THREAD];
  float3 color_accum[PIXELS_PER_THREAD];

  const int in_tile_x = threadIdx.x % TILE_SIZE_BWD;                     // local tile x
  const int in_tile_y = threadIdx.x / TILE_SIZE_BWD * PIXELS_PER_THREAD; // local tile y

  const int base_pixel_x = (tile_idx % num_tiles_x) * TILE_SIZE_BWD + in_tile_x;
  const int base_pixel_y = (tile_idx / num_tiles_x) * TILE_SIZE_BWD + in_tile_y;

  int index_in_tile = 0;
  bool valid_pixel = false;
#pragma unroll
  for (int i = 0; i < PIXELS_PER_THREAD; i++) {
    const int global_pixel_x = base_pixel_x;
    const int global_pixel_y = base_pixel_y + i;

    valid_pixel = global_pixel_x < image_width && global_pixel_y < image_height;
    if (valid_pixel) {
      const int pixel_id = global_pixel_y * image_width + global_pixel_x;
      _splats_per_pixel[i][threadIdx.y * blockDim.x + threadIdx.x] = num_splats_per_pixel[pixel_id];
      _trans_final[i][threadIdx.y * blockDim.x + threadIdx.x] = final_weight_per_pixel[pixel_id];
#pragma unroll
      for (int channel = 0; channel < 3; channel++) {
        _image_grad[channel][i][threadIdx.y * blockDim.x + threadIdx.x] = grad_image[pixel_id * 3 + channel];
      }
      index_in_tile = max(index_in_tile, _splats_per_pixel[i][threadIdx.y * blockDim.x + threadIdx.x]);
      // Init per-pixel values
      T[i] = _trans_final[i][threadIdx.y * blockDim.x + threadIdx.x];
    } else {
      T[i] = 0.0f;
      _splats_per_pixel[i][threadIdx.y * blockDim.x + threadIdx.x] = 0;
    }
    T_final[i] = T[i];
    color_accum[i] = {0.0f, 0.0f, 0.0f};
  }
  index_in_tile = cg::reduce(warp, index_in_tile, cg::greater<int>()) - 1; // max depth in tile

  assert(index_in_tile < splat_idx_end);

  // Starting index in sorted splats
  const int *splats_in_tile = &gaussian_idx_by_splat_idx[splat_idx_start];

  // Iterate on splats in tile back to front
  for (; index_in_tile >= 0; index_in_tile--) {
    const int gaussian_idx = splats_in_tile[index_in_tile];

    float basic;
    float linear;
    float quad;
    float2 d = {0.0f, 0.0f};

    d.x = uvs[gaussian_idx * 2 + 0] - (float)base_pixel_x;
    d.y = uvs[gaussian_idx * 2 + 1] - (float)base_pixel_y;
    const float inv_cov00 = conic[gaussian_idx * 3 + 0];
    const float inv_cov01 = conic[gaussian_idx * 3 + 1];
    const float inv_cov11 = conic[gaussian_idx * 3 + 2];
    basic = -0.5f * (inv_cov00 * d.x * d.x + 2.0f * inv_cov01 * d.x * d.y + inv_cov11 * d.y * d.y);
    linear = inv_cov11 * d.y + inv_cov01 * d.x;
    quad = -0.5f * inv_cov11;

    float3 color = {rgb[gaussian_idx * 3 + 0], rgb[gaussian_idx * 3 + 1], rgb[gaussian_idx * 3 + 2]};
    float opa = 1.0f / (1.0f + __expf(-opacity[gaussian_idx]));

    // Zero gradients for current gaussian
    float3 grad_rgb_tile = {0.0f, 0.0f, 0.0f};
    float grad_opacity_tile = 0.0f;
    float grad_basic = 0.0f;
    float grad_linear = 0.0f;
    float grad_quad = 0.0f;

    // Raster scan on all pixels per thread
#pragma unroll
    for (int i = 0; i < PIXELS_PER_THREAD; i++) {
      const int global_pixel_x = base_pixel_x;
      const int global_pixel_y = base_pixel_y + i;

      valid_pixel = global_pixel_x < image_width && global_pixel_y < image_height;

      float power = fminf(0.0f, basic + linear * i + quad * i * i);
      float g = __expf(power);
      float alpha = min(0.99f, opa * g);

      // Mask out low alpha and depth
      bool valid_splat = valid_pixel;
      valid_splat &= (alpha >= 0.00392156862f);
      valid_splat &= (index_in_tile <= _splats_per_pixel[i][threadIdx.y * blockDim.x + threadIdx.x]);

      const unsigned int valid_mask = __any_sync(0xFFFFFFFF, valid_splat);

      if (valid_mask) {
        alpha *= valid_splat;
        g *= valid_splat;

        T[i] *= 1.0f / (1.0f - alpha);

        // RGB gradients
        grad_rgb_tile.x += alpha * T[i] * _image_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x];
        grad_rgb_tile.y += alpha * T[i] * _image_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x];
        grad_rgb_tile.z += alpha * T[i] * _image_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x];

        float grad_alpha = 0.0f;
        // alpha gradient
        grad_alpha += (color.x - color_accum[i].x) * _image_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x];
        grad_alpha += (color.y - color_accum[i].y) * _image_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x];
        grad_alpha += (color.z - color_accum[i].z) * _image_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x];
        grad_alpha *= T[i];

        // account for background contribution
        float bg_dot_pixel = 0;
        bg_dot_pixel += background_opacity * _image_grad[0][i][threadIdx.y * blockDim.x + threadIdx.x];
        bg_dot_pixel += background_opacity * _image_grad[1][i][threadIdx.y * blockDim.x + threadIdx.x];
        bg_dot_pixel += background_opacity * _image_grad[2][i][threadIdx.y * blockDim.x + threadIdx.x];
        grad_alpha += (-T_final[i] / (1.0f - alpha)) * bg_dot_pixel;

        // opacity gradient
        grad_opacity_tile += g * grad_alpha * opa * (1.0f - opa);

        color_accum[i].x = alpha * color.x + (1.0f - alpha) * color_accum[i].x;
        color_accum[i].y = alpha * color.y + (1.0f - alpha) * color_accum[i].y;
        color_accum[i].z = alpha * color.z + (1.0f - alpha) * color_accum[i].z;

        // G gradient
        const float grad_g = grad_alpha * opa;
        const float grad_power = g * grad_g;

        grad_basic += grad_power;
        grad_linear += grad_power * i;
        grad_quad += grad_power * i * i;
      }
    }
    // Accumulate gradients across tile
    if (__any_sync(0xFFFFFFFF, grad_opacity_tile != 0.0f)) {
      // RGB
      grad_rgb_tile.x = cg::reduce(warp, grad_rgb_tile.x, cg::plus<float>());
      grad_rgb_tile.y = cg::reduce(warp, grad_rgb_tile.y, cg::plus<float>());
      grad_rgb_tile.z = cg::reduce(warp, grad_rgb_tile.z, cg::plus<float>());

      // Opacity
      grad_opacity_tile = cg::reduce(warp, grad_opacity_tile, cg::plus<float>());

      // UV
      float grad_u_tile = 0.0f;
      float grad_v_tile = 0.0f;

      grad_u_tile = (-inv_cov00 * d.x - inv_cov01 * d.y) * grad_basic + inv_cov01 * grad_linear;
      grad_v_tile = (-inv_cov11 * d.y - inv_cov01 * d.x) * grad_basic + inv_cov11 * grad_linear;

      grad_u_tile *= 0.5f * image_width;
      grad_v_tile *= 0.5f * image_height;

      grad_u_tile = cg::reduce(warp, grad_u_tile, cg::plus<float>());
      grad_v_tile = cg::reduce(warp, grad_v_tile, cg::plus<float>());

      // Conic
      float3 grad_conic_tile = {0.0f, 0.0f, 0.0f};

      const float grad_inv_cov00 = grad_basic * (-0.5f * d.x * d.x);
      const float grad_inv_cov11 = grad_basic * (-0.5f * d.y * d.y) + (grad_linear * d.y) - (0.5f * grad_quad);
      const float grad_inv_cov01 = grad_basic * (-d.x * d.y) + grad_linear * d.x;

      grad_conic_tile.x = grad_inv_cov00;
      grad_conic_tile.y = grad_inv_cov01;
      grad_conic_tile.z = grad_inv_cov11;

      grad_conic_tile.x = cg::reduce(warp, grad_conic_tile.x, cg::plus<float>());
      grad_conic_tile.y = cg::reduce(warp, grad_conic_tile.y, cg::plus<float>());
      grad_conic_tile.z = cg::reduce(warp, grad_conic_tile.z, cg::plus<float>());

      if (warp.thread_rank() == 0) {
        atomicAdd(&grad_rgb[gaussian_idx * 3 + 0], grad_rgb_tile.x);
        atomicAdd(&grad_rgb[gaussian_idx * 3 + 1], grad_rgb_tile.y);
        atomicAdd(&grad_rgb[gaussian_idx * 3 + 2], grad_rgb_tile.z);

        atomicAdd(&grad_opacity[gaussian_idx], grad_opacity_tile);

        atomicAdd(&grad_conic[gaussian_idx * 3 + 0], grad_conic_tile.x);
        atomicAdd(&grad_conic[gaussian_idx * 3 + 1], grad_conic_tile.y);
        atomicAdd(&grad_conic[gaussian_idx * 3 + 2], grad_conic_tile.z);

        atomicAdd(&grad_uv[gaussian_idx * 2 + 0], grad_u_tile);
        atomicAdd(&grad_uv[gaussian_idx * 2 + 1], grad_v_tile);
      }
    }
  }
}

void render_image_backward(const float *const uvs, const float *const opacity, const float *const conic,
                           const float *const rgb, const float background_opacity, const int *const sorted_splats,
                           const int *const splat_range_by_tile, const int *const num_splats_per_pixel,
                           const float *const final_weight_per_pixel, const float *const grad_image,
                           const int image_width, const int image_height, float *grad_rgb, float *grad_opacity,
                           float *grad_uv, float *grad_conic, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(uvs);
  ASSERT_DEVICE_POINTER(opacity);
  ASSERT_DEVICE_POINTER(conic);
  ASSERT_DEVICE_POINTER(rgb);
  ASSERT_DEVICE_POINTER(sorted_splats);
  ASSERT_DEVICE_POINTER(splat_range_by_tile);
  ASSERT_DEVICE_POINTER(num_splats_per_pixel);
  ASSERT_DEVICE_POINTER(final_weight_per_pixel);
  ASSERT_DEVICE_POINTER(grad_image);
  ASSERT_DEVICE_POINTER(grad_rgb);
  ASSERT_DEVICE_POINTER(grad_opacity);
  ASSERT_DEVICE_POINTER(grad_uv);
  ASSERT_DEVICE_POINTER(grad_conic);

  const int tiles_per_block = 4;
  const int num_tiles_x = (image_width + TILE_SIZE_BWD - 1) / TILE_SIZE_BWD;
  const int num_tiles_y = (image_height + TILE_SIZE_BWD - 1) / TILE_SIZE_BWD;
  const int total_blocks = (num_tiles_x * num_tiles_y + tiles_per_block - 1) / tiles_per_block;

  dim3 block_size(32, tiles_per_block, 1);
  dim3 grid_size(total_blocks, 1, 1);

  // Launch the single, non-templated kernel
  render_tiles_backward_kernel<<<grid_size, block_size, 0, stream>>>(
      num_tiles_x, num_tiles_y, uvs, opacity, rgb, conic, splat_range_by_tile, sorted_splats, background_opacity,
      num_splats_per_pixel, final_weight_per_pixel, grad_image, image_width, image_height, grad_rgb, grad_opacity,
      grad_uv, grad_conic);
}
