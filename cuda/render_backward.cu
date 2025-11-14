// render_backward.cu

#include "checks.cuh"
#include "gsplat_cuda/cuda_backward.cuh"
#include <__clang_cuda_builtin_vars.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <thread>

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
  // Tile outside of image
  if (tile_idx >= num_tiles_x * num_tiles_y)
    return;

  const int splat_idx_start = splat_start_end_idx_by_tile_idx[tile_idx];
  const int splat_idx_end = splat_start_end_idx_by_tile_idx[tile_idx + 1];
  int num_splats_this_tile = splat_idx_end - splat_idx_start;

  __shared__ float _image_grad[3][PIXELS_PER_THREAD][32];
  __shared__ float _trans_final[PIXELS_PER_THREAD][32];
  __shared__ unsigned int _splats_per_pixel[PIXELS_PER_THREAD][32];

  // Per-pixel variables stored in registers
  bool background_initialized[PIXELS_PER_THREAD];
  float T[PIXELS_PER_THREAD];
  float3 color_accum[PIXELS_PER_THREAD];

  const int in_tile_x = threadIdx.x % TILE_SIZE_BWD;                     // local tile x
  const int in_tile_y = threadIdx.x / TILE_SIZE_BWD * PIXELS_PER_THREAD; // local tile y

  const int base_pixel_x = tile_idx % num_tiles_x * TILE_SIZE_BWD + in_tile_x;
  const int base_pixel_y = tile_idx / num_tiles_x * TILE_SIZE_BWD + in_tile_y;

  unsigned int index_in_tile = 0;
#pragma unroll
  for (int i = 0; i < PIXELS_PER_THREAD; i++) {
    const int global_pixel_x = base_pixel_x;
    const int global_pixel_y = base_pixel_y + i;

    bool valid_pixel = global_pixel_x < image_width && global_pixel_y < image_height;
    if (valid_pixel) {
      const int pixel_id = global_pixel_y * image_width + global_pixel_x;
      _splats_per_pixel[i][threadIdx.x] = num_splats_per_pixel[pixel_id];
      _trans_final[i][threadIdx.x] = final_weight_per_pixel[pixel_id];
#pragma unroll
      for (int channel = 0; channel < 3; channel++) {
        _image_grad[channel][i][threadIdx.x] = grad_image[pixel_id * 3 + channel];
      }
    }
    // Init per-pixel values
    background_initialized[i] = false;
    color_accum[i] = {0.0f, 0.0f, 0.0f};
    T[i] = _trans_final[i][threadIdx.x];

    index_in_tile = max(index_in_tile, _splats_per_pixel[i][threadIdx.x]);
  }
  index_in_tile = __reduce_max_sync(0xFFFFFFFF, index_in_tile); // max depth in tile

  // Starting index in sorted splats
  const int *splats_in_tile = &gaussian_idx_by_splat_idx[splat_idx_start];

  // Iterate on splats in tile back to front
  for (; index_in_tile >= 0; index_in_tile--) {
    const int gaussian_idx = splats_in_tile[index_in_tile];

    float basic;
    float linear;
    float quad;
    float inv_det;
    float2 d = {0.0f, 0.0f};
    {
      d.x = (float)base_pixel_x - uvs[gaussian_idx * 2 + 0];
      d.y = (float)base_pixel_y - uvs[gaussian_idx * 2 + 1];
      const float a = conic[gaussian_idx * 3 + 0] + 0.3f;
      const float b = conic[gaussian_idx * 3 + 1];
      const float c = conic[gaussian_idx * 3 + 2] + 0.3f;
      inv_det = 1.0f / a * c - b * b;
      basic = -0.5f * (a * d.x * d.x + c * d.y * d.y + 2.0f * b * d.x * d.y);
      linear = b * d.x - c * d.y;
      quad = -0.5f * c;
    }

    float3 color = {rgb[gaussian_idx * 3 + 0], rgb[gaussian_idx * 3 + 1], rgb[gaussian_idx * 3 + 2]};
    float opa = opacity[gaussian_idx];

    // Zero gradients for current gaussian
    float3 grad_rgb = {0.0f, 0.0f, 0.0f};
    float grad_alpha = 0.0f;
    float grad_basic = 0.0f;
    float grad_bxcy = 0.0f;
    float grad_neg_half_c = 0.0f;

    // Raster scan on all pixels per thread
#pragma unroll
    for (int i = 0; i < PIXELS_PER_THREAD; i++) {
      float power = basic + linear * i + quad * i * i;
      float g = __expf(power * inv_det);
      float alpha = min(0.99f, opa * g);

      // Mask out low alpha and depth
      const bool valid_splat = (alpha >= 0.00392156862f) && (index_in_tile <= _splats_per_pixel[i][threadIdx.x]);
      const unsigned int valid_mask = __ballot_sync(valid_mask, valid_splat);

      if (__any_sync(0xFFFFFFFF, valid_mask != 0)) {
        // Initialize background color
        if (!background_initialized[i]) {
          const float background_weight = 1.0f - (alpha * T[i] + 1.0f - T[i]);
          if (background_weight > 0.001f) {
            color_accum[i].x += background_opacity * background_weight;
            color_accum[i].y += background_opacity * background_weight;
            color_accum[i].z += background_opacity * background_weight;
          }
          background_initialized[i] = true;
        } else {
          // alpha reciprical
          float ra = 1.0f / (1.0f - alpha);
          T[i] *= ra;
        }

        const float fac = alpha * T[i];

        // RGB gradients
        grad_rgb.x += fac * _image_grad[0][i][threadIdx.x];
        grad_rgb.y += fac * _image_grad[1][i][threadIdx.x];
        grad_rgb.z += fac * _image_grad[2][i][threadIdx.x];

        // alpha gradient
        grad_alpha += (rgb[gaussian_idx * 3 + 0] - color_accum[i].x) * T[i] * _image_grad[0][i][threadIdx.x];
        grad_alpha += (rgb[gaussian_idx * 3 + 1] - color_accum[i].y) * T[i] * _image_grad[1][i][threadIdx.x];
        grad_alpha += (rgb[gaussian_idx * 3 + 2] - color_accum[i].z) * T[i] * _image_grad[2][i][threadIdx.x];
        color_accum[i].x += alpha * (rgb[gaussian_idx * 3 + 0] - color_accum[i].x);
        color_accum[i].y += alpha * (rgb[gaussian_idx * 3 + 1] - color_accum[i].y);
        color_accum[i].z += alpha * (rgb[gaussian_idx * 3 + 2] - color_accum[i].z);
      }
    }
  }
  /*
  // Coopertive group at block level i.e tiles
  cg::thread_block tile_thread_group = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(tile_thread_group);
  const int warp_rank = warp.meta_group_rank();


    int chunk_start = chunk_idx * CHUNK_SIZE;
    int chunk_end = min((chunk_idx + 1) * CHUNK_SIZE, num_splats_this_tile);
    for (int i = chunk_end - chunk_start - 1; i >= 0; i--) {
      const int tile_splat_idx = chunk_start + i;

      float grad_opa = 0.0f, grad_u = 0.0f, grad_v = 0.0f;
      float grad_rgb_local[3] = {0.0f, 0.0f, 0.0f};
      float grad_conic_splat[3] = {0.0f, 0.0f, 0.0f};

      if (valid_pixel && tile_splat_idx < num_splats_this_pixel) {
        const float u_mean = _uvs[i * 2 + 0];
        const float v_mean = _uvs[i * 2 + 1];
        const float u_diff = float(u_splat) - u_mean;
        const float v_diff = float(v_splat) - v_mean;

        const float a = _conic[i * 3 + 0] + 0.3f;
        const float b = _conic[i * 3 + 1];
        const float c = _conic[i * 3 + 2] + 0.3f;

        const float det = a * c - b * b;
        const float reciprocal_det = 1.0f / det;
        const float mh_sq = (c * u_diff * u_diff - 2.0f * b * u_diff * v_diff + a * v_diff * v_diff) * reciprocal_det;

        float g = 0.0f;
        if (mh_sq > 0.0f) {
          g = __expf(-0.5f * mh_sq);
        }

        // effective opacity
        float alpha = fminf(0.99f, _opacity[i] * g);

        // Gaussian does not contribute to image
        if (alpha >= 0.00392156862f) {
          if (!background_initialized) {
            const float background_weight = 1.0f - (alpha * T + 1.0f - T);
            if (background_weight > 0.001f) {
#pragma unroll
              for (int j = 0; j < 3; j++) {
                color_accum[j] += background_opacity * background_weight;
              }
            }
            background_initialized = true;
          }

          // alpha reciprical
          float ra = 1.0f / (1.0f - alpha);

          // update T inlu if not first iteration
          if (i < num_splats_this_pixel - 1)
            T *= ra;

          const float fac = alpha * T;

#pragma unroll
          for (int j = 0; j < 3; j++)
            grad_rgb_local[j] = fac * grad_image_local[j];

          float grad_alpha = 0.0f;
#pragma unroll
          for (int j = 0; j < 3; j++)
            grad_alpha += (_rgb[i * 3 + j] * T - color_accum[j] * ra) * grad_image_local[j];

          // opacity grad
          grad_opa = g * grad_alpha;
          // sigmoid gradient
          grad_opa *= _opacity[i] * (1.0f - _opacity[i]);

          // gradient of gaussian probability
          const float grad_g = grad_alpha * _opacity[i];
          const float grad_mh_sq = -0.5f * g * grad_g;

          // Normalize viewspace positional gradients (-1 to 1)
          const float u_norm = image_width * 0.5f;
          const float v_norm = image_height * 0.5f;
          // UV gradients
          grad_u = -(-b * v_diff - b * v_diff + 2.0f * c * u_diff) * reciprocal_det * grad_mh_sq * u_norm;
          grad_v = -(2.0f * a * v_diff - b * u_diff - b * u_diff) * reciprocal_det * grad_mh_sq * v_norm;

          // Conic gradients
          const float common_frac =
              (a * v_diff * v_diff - b * u_diff * v_diff - b * u_diff * v_diff + c * u_diff * u_diff) * reciprocal_det *
              reciprocal_det;
          grad_conic_splat[0] = (-c * common_frac + v_diff * v_diff * reciprocal_det) * grad_mh_sq;
          grad_conic_splat[1] = (b * common_frac - u_diff * v_diff * reciprocal_det) * grad_mh_sq;
          grad_conic_splat[2] = (-a * common_frac + u_diff * u_diff * reciprocal_det) * grad_mh_sq;

          // Update color accum
#pragma unroll
          for (int j = 0; j < 3; j++)
            color_accum[j] += _rgb[i * 3 + j] * fac;
        }
      }

      // --- Stage 1: Warp-Level Reduction ---
      // Each warp reduces its 32 threads' values. Result lands in thread 0 of the warp.
      grad_opa = cg::reduce(warp, grad_opa, cg::plus<float>());
      grad_u = cg::reduce(warp, grad_u, cg::plus<float>());
      grad_v = cg::reduce(warp, grad_v, cg::plus<float>());
#pragma unroll
      for (int j = 0; j < 3; j++)
        grad_conic_splat[j] = cg::reduce(warp, grad_conic_splat[j], cg::plus<float>());
#pragma unroll
      for (int j = 0; j < 3; j++)
        grad_rgb_local[j] = cg::reduce(warp, grad_rgb_local[j], cg::plus<float>());

      // --- Stage 2: Write Warp Sums to Shared Memory ---
      // Thread 0 of each warp writes its sum to the intermediate buffer.
      if (warp.thread_rank() == 0) {
        _grad_opa[warp_rank] = grad_opa;
        _grad_u[warp_rank] = grad_u;
        _grad_v[warp_rank] = grad_v;
#pragma unroll
        for (int j = 0; j < 3; j++)
          _grad_rgb[warp_rank * 3 + j] = grad_rgb_local[j];
#pragma unroll
        for (int j = 0; j < 3; j++)
          _grad_conic[warp_rank * 3 + j] = grad_conic_splat[j];
      }

      // Sync block to ensure all warp sums are in shared memory
      tile_thread_group.sync();

      // --- Stage 3: Block-Level Reduction (Reduce Warp Sums) ---
      // A single warp (warp 0) now reduces the intermediate sums.
      if (warp_rank == 0) {
        // Load intermediate sums into warp 0's registers
        float final_grad_opa = (warp.thread_rank() < REDUCE_SIZE) ? _grad_opa[warp.thread_rank()] : 0.0f;
        float final_grad_u = (warp.thread_rank() < REDUCE_SIZE) ? _grad_u[warp.thread_rank()] : 0.0f;
        float final_grad_v = (warp.thread_rank() < REDUCE_SIZE) ? _grad_v[warp.thread_rank()] : 0.0f;
        float final_grad_rgb[3] = {0.0f, 0.0f, 0.0f};
        float final_grad_conic[3] = {0.0f, 0.0f, 0.0f};

        if (warp.thread_rank() < REDUCE_SIZE) {
#pragma unroll
          for (int j = 0; j < 3; j++)
            final_grad_rgb[j] = _grad_rgb[warp.thread_rank() * 3 + j];
#pragma unroll
          for (int j = 0; j < 3; j++)
            final_grad_conic[j] = _grad_conic[warp.thread_rank() * 3 + j];
        }

        // Reduce within warp 0
        final_grad_opa = cg::reduce(warp, final_grad_opa, cg::plus<float>());
        final_grad_u = cg::reduce(warp, final_grad_u, cg::plus<float>());
        final_grad_v = cg::reduce(warp, final_grad_v, cg::plus<float>());
#pragma unroll
        for (int j = 0; j < 3; j++)
          final_grad_rgb[j] = cg::reduce(warp, final_grad_rgb[j], cg::plus<float>());
#pragma unroll
        for (int j = 0; j < 3; j++)
          final_grad_conic[j] = cg::reduce(warp, final_grad_conic[j], cg::plus<float>());

        // --- Stage 4: Write Final Sum ---
        // Only thread 0 of warp 0 writes the final result for this Gaussian.
        if (warp.thread_rank() == 0) {
          _grad_opa_chunk[i] = final_grad_opa;
          _grad_u_chunk[i] = final_grad_u;
          _grad_v_chunk[i] = final_grad_v;
#pragma unroll
          for (int j = 0; j < 3; j++)
            _grad_rgb_chunk[i * 3 + j] = final_grad_rgb[j];
#pragma unroll
          for (int j = 0; j < 3; j++)
            _grad_conic_chunk[i * 3 + j] = final_grad_conic[j];
        }
      }

      // Sync to ensure chunk buffers are written before the next 'i' iteration
      tile_thread_group.sync();
    }

    const int num_splats_in_chunk = chunk_end - chunk_start;
    for (int i = thread_id; i < num_splats_in_chunk; i += block_size) {
      const int global_write_idx = splat_idx_start + chunk_start + i;

      grad_opacity[global_write_idx] = _grad_opa_chunk[i];
      grad_uv[global_write_idx * 2 + 0] = _grad_u_chunk[i];
      grad_uv[global_write_idx * 2 + 1] = _grad_v_chunk[i];
#pragma unroll
      for (int j = 0; j < 3; j++)
        grad_rgb[global_write_idx * 3 + j] = _grad_rgb_chunk[i * 3 + j];
#pragma unroll
      for (int j = 0; j < 3; j++)
        grad_conic[global_write_idx * 3 + j] = _grad_conic_chunk[i * 3 + j];
    }
  }
  */
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
