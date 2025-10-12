#include "checks.cuh"
#include "gsplat/cuda_forward.hpp"
#include <cub/cub.cuh>
#include <math_constants.h>

__device__ __forceinline__ bool z_distance_culling(const float z, const float near_thresh, const float far_thresh) {
  return z >= near_thresh && z <= far_thresh;
}
__device__ __forceinline__ bool frustum_culling(const float u, const float v, const int padding, const int width,
                                                const int height) {
  return u >= (-1 * padding) && u <= width + padding && v >= (-1 * padding) && v <= height + padding;
}

__global__ void frustum_culling_kernel(const float *__restrict__ uv, const float *__restrict__ xyz, const int N,
                                       const float near_thresh, const float far_thresh, const int padding,
                                       const int width, const int height, bool *mask) {
  constexpr int XYZ_STRIDE = 3;
  constexpr int UV_STRIDE = 2;

  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= N) {
    return;
  }

  const float u = uv[i * UV_STRIDE + 0];
  const float v = uv[i * UV_STRIDE + 1];

  const float z = xyz[i * XYZ_STRIDE + 2];

  mask[i] = z_distance_culling(z, near_thresh, far_thresh) && frustum_culling(u, v, padding, width, height);
}

__device__ __forceinline__ bool
split_axis_test(const float *__restrict__ obb,        // [tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y]
                const float *__restrict__ tile_bounds // [left, right, top, bottom]
) {
  // axis0 - X axis
  const float obb_min_x = fminf(fminf(obb[0], obb[2]), fminf(obb[4], obb[6]));
  const float obb_max_x = fmaxf(fmaxf(obb[0], obb[2]), fmaxf(obb[4], obb[6]));
  if (obb_min_x > tile_bounds[1] || obb_max_x < tile_bounds[0])
    return false;

  // axis1 - Y axis
  const float obb_min_y = fminf(fminf(obb[1], obb[3]), fminf(obb[5], obb[7]));
  const float obb_max_y = fmaxf(fmaxf(obb[1], obb[3]), fmaxf(obb[5], obb[7]));
  if (obb_min_y > tile_bounds[3] || obb_max_y < tile_bounds[2])
    return false;

  // axis 2 - obb major axis
  const float obb_major_axis_x = obb[2] - obb[0];
  const float obb_major_axis_y = obb[3] - obb[1];
  float tl_ax2 = obb_major_axis_x * tile_bounds[0] + obb_major_axis_y * tile_bounds[2];
  float tr_ax2 = obb_major_axis_x * tile_bounds[1] + obb_major_axis_y * tile_bounds[2];
  float bl_ax2 = obb_major_axis_x * tile_bounds[0] + obb_major_axis_y * tile_bounds[3];
  float br_ax2 = obb_major_axis_x * tile_bounds[1] + obb_major_axis_y * tile_bounds[3];
  float min_tile = fminf(fminf(tl_ax2, tr_ax2), fminf(bl_ax2, br_ax2));
  float max_tile = fmaxf(fmaxf(tl_ax2, tr_ax2), fmaxf(bl_ax2, br_ax2));
  const float obb_r_ax2 = obb_major_axis_x * obb[2] + obb_major_axis_y * obb[3];
  const float obb_l_ax2 = obb_major_axis_x * obb[0] + obb_major_axis_y * obb[1];
  float min_obb = fminf(obb_r_ax2, obb_l_ax2);
  float max_obb = fmaxf(obb_r_ax2, obb_l_ax2);
  if (min_tile > max_obb || max_tile < min_obb)
    return false;

  // axis 3 - obb minor axis
  const float obb_minor_axis_x = obb[2] - obb[6];
  const float obb_minor_axis_y = obb[3] - obb[7];
  tl_ax2 = obb_minor_axis_x * tile_bounds[0] + obb_minor_axis_y * tile_bounds[2];
  tr_ax2 = obb_minor_axis_x * tile_bounds[1] + obb_minor_axis_y * tile_bounds[2];
  bl_ax2 = obb_minor_axis_x * tile_bounds[0] + obb_minor_axis_y * tile_bounds[3];
  br_ax2 = obb_minor_axis_x * tile_bounds[1] + obb_minor_axis_y * tile_bounds[3];
  min_tile = fminf(fminf(tl_ax2, tr_ax2), fminf(bl_ax2, br_ax2));
  max_tile = fmaxf(fmaxf(tl_ax2, tr_ax2), fmaxf(bl_ax2, br_ax2));
  const float obb_t_ax2 = obb_minor_axis_x * obb[2] + obb_minor_axis_y * obb[3];
  const float obb_b_ax2 = obb_minor_axis_x * obb[6] + obb_minor_axis_y * obb[7];
  min_obb = fminf(obb_t_ax2, obb_b_ax2);
  max_obb = fmaxf(obb_t_ax2, obb_b_ax2);
  if (min_tile > max_obb || max_tile < min_obb)
    return false;

  return true;
}

__device__ __forceinline__ int compute_obb(const float u, const float v, const float a, const float b, const float c,
                                           const float mh_dist, float *obb) {
  const float t_sum = a + c;
  const float t_diff = a - c;
  const float discriminant = t_diff * t_diff + 4.f * b * b;
  const float root = sqrtf(discriminant);      // Guaranteed non-negative
  const float lambda1 = 0.5f * (t_sum + root); // Major eigenvalue
  const float lambda2 = 0.5f * (t_sum - root); // Minor eigenvalue

  const float r_major = mh_dist * sqrtf(fmaxf(0.f, lambda1));
  const float r_minor = mh_dist * sqrtf(fmaxf(0.f, lambda2));

  float cos_theta, sin_theta;
  if (fabsf(root) < 1e-7f) {
    // Handle the case of a circle (a=c, b=0), where rotation is arbitrary.
    cos_theta = 1.f;
    sin_theta = 0.f;
  } else {
    // Use half-angle trigonometric identities:
    // cos^2(t) = (1 + cos(2t))/2, sin^2(t) = (1 - cos(2t))/2
    // where cos(2t) = (a-c)/root and sin(2t) = 2b/root.
    const float inv_root = 1.f / root;
    const float cos2theta = t_diff * inv_root;

    cos_theta = sqrtf(0.5f * (1.f + cos2theta));
    sin_theta = sqrtf(0.5f * (1.f - cos2theta));

    // The sign of sin(theta) is the same as the sign of b.
    sin_theta = copysignf(sin_theta, b);
  }

  // Calculate the two orthogonal vectors defining the OBB's orientation and size
  const float v1_x = r_major * cos_theta;
  const float v1_y = r_major * sin_theta;
  const float v2_x = -r_minor * sin_theta;
  const float v2_y = r_minor * cos_theta;

  // Compute the 4 OBB corners by adding/subtracting vectors from the center
  obb[0] = u - v1_x - v2_x; // Bottom-left corner
  obb[1] = v - v1_y - v2_y;
  obb[2] = u + v1_x - v2_x; // Bottom-right corner
  obb[3] = v + v1_y - v2_y;
  obb[4] = u - v1_x + v2_x; // Top-left corner
  obb[5] = v - v1_y + v2_y;
  obb[6] = u + v1_x + v2_x; // Top-right corner
  obb[7] = v + v1_y + v2_y;

  return ceilf(r_major * 0.0625f) + 1; // Use multiplication for division by 16
}

__device__ __forceinline__ int get_write_index(const bool write, const int lane, const unsigned int active_mask,
                                               int *global_index) {
  // Get mask of threads that want to write
  unsigned mask = __ballot_sync(active_mask, write);
  // no threads in warp will write
  if (mask == 0) {
    return -1;
  }
  // Count how many threads before want to write
  int prefix = __popc(mask & ((1u << lane) - 1));

  // First active thread in warp gets block of slots
  int base_slot = 0;
  int leader_lane = __ffs(mask) - 1;
  if (lane == leader_lane) {
    base_slot = atomicAdd(global_index, __popc(mask));
  }

  // Broadcast base slot to all threads
  base_slot = __shfl_sync(active_mask, base_slot, leader_lane);

  return write ? (base_slot + prefix) : -1;
}

__device__ __forceinline__ int warpSum(unsigned mask, int val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(mask, val, offset);
  return val;
}

__global__ void coarse_binning_kernel(const float *__restrict__ uvs, const float *__restrict__ conic,
                                      const float mh_dist, const int n_tiles_x, const int n_tiles_y, const int N,
                                      int *buffer_bytes, int2 *pairs, int *global_index) {
  const int gaussian_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // mask active threads for warpSum
  unsigned active_mask = __ballot_sync(0xFFFFFFFF, gaussian_idx < N);

  if (gaussian_idx >= N)
    return;

  const float u = uvs[gaussian_idx * 2];
  const float v = uvs[gaussian_idx * 2 + 1];
  const float a = conic[gaussian_idx * 3] + 0.25f;
  const float b = conic[gaussian_idx * 3 + 1] / 2.0f;
  const float c = conic[gaussian_idx * 3 + 2] + 0.25f;

  float obb[8];
  const int radius_tiles = compute_obb(u, v, a, b, c, mh_dist, obb);

  // BUG Check the bounds on this
  const int projected_tile_x = floorf(u / 16.0f);
  const int start_tile_x = max(0, projected_tile_x - radius_tiles);
  const int end_tile_x = min(n_tiles_x, projected_tile_x + radius_tiles + 1);
  const int projected_tile_y = floorf(v / 16.0f);
  const int start_tile_y = max(0, projected_tile_y - radius_tiles);
  const int end_tile_y = min(n_tiles_y, projected_tile_y + radius_tiles + 1);

  // clamp negative values
  const int num_x_tiles = max(0, end_tile_x - start_tile_x);
  const int num_y_tiles = max(0, end_tile_y - start_tile_y);
  const int num_pairs_for_thread = num_x_tiles * num_y_tiles;

  // Get required bytes for buffer
  if (pairs == nullptr) {
    const int lane_id = gaussian_idx & 0x1f;
    const int warp_sum = warpSum(active_mask, num_pairs_for_thread);
    if (lane_id == 0) {
      atomicAdd(buffer_bytes, warp_sum * sizeof(int));
    }
  } else {
    // Write pairs to buffer
    int write_offset = atomicAdd(global_index, num_pairs_for_thread);
    int curr_pair_id = 0;
    for (int tile_x = start_tile_x; tile_x < end_tile_x; tile_x++) {
      for (int tile_y = start_tile_y; tile_y < end_tile_y; tile_y++) {
        int2 pair = {tile_y * n_tiles_x + tile_x, gaussian_idx};
        pairs[write_offset + curr_pair_id] = pair;
        curr_pair_id++;
      }
    }
  }
}

__global__ void generate_splats_kernel(const float *__restrict__ uvs, const float *__restrict__ xyz_camera_frame,
                                       const float *__restrict__ conic, const int2 *__restrict__ pairs,
                                       const float mh_dist, const int num_pairs, const int num_tiles_x,
                                       const int num_tiles_y, float *max_z, int *gaussian_idx_by_splat_idx,
                                       double *sort_keys, int *global_splat_counter) {
  int pair_id = blockIdx.x * blockDim.x + threadIdx.x;

  // Mask of all active threads
  unsigned active_mask = __ballot_sync(0xFFFFFFFF, pair_id < num_pairs);

  if (pair_id >= num_pairs)
    return;

  // load pair data
  int2 pair_data = pairs[pair_id];
  const int tile_idx = pair_data.x;
  const int gaussian_idx = pair_data.y;

  // get Guassian parameters
  const float u = uvs[gaussian_idx * 2];
  const float v = uvs[gaussian_idx * 2 + 1];
  const double z = (double)(xyz_camera_frame[gaussian_idx * 3 + 2]);
  const float a = conic[gaussian_idx * 3] + 0.25f;
  const float b = conic[gaussian_idx * 3 + 1] / 2.0f;
  const float c = conic[gaussian_idx * 3 + 2] + 0.25f;

  float obb[8];
  const int radius_tiles = compute_obb(u, v, a, b, c, mh_dist, obb);

  const int tile_x = tile_idx % num_tiles_x;
  const int tile_y = tile_idx / num_tiles_x;

  double tile_idx_key_multiplier = 0.0;
  tile_idx_key_multiplier = *max_z + 1.0f;

  float tile_bounds[4];
  tile_bounds[0] = __int2float_rn(tile_x) * 16.0f;
  tile_bounds[1] = __int2float_rn(tile_x + 1) * 16.0f;
  tile_bounds[2] = __int2float_rn(tile_y) * 16.0f;
  tile_bounds[3] = __int2float_rn(tile_y + 1) * 16.0f;

  bool intersects = false;
  intersects = split_axis_test(obb, tile_bounds);

  // get position of splat in global array
  const int lane_id = pair_id & 0x1f;
  int splat_idx = get_write_index(intersects, lane_id, active_mask, global_splat_counter);

  if (intersects) {
    gaussian_idx_by_splat_idx[splat_idx] = gaussian_idx;
    sort_keys[splat_idx] = z + tile_idx_key_multiplier * __int2double_rn(tile_idx);
  }
}

__global__ void find_tile_boundaries_kernel(const double *__restrict__ sorted_keys, const int num_splats, float *max_z,
                                            int *__restrict__ splat_start_end_idx_by_tile_idx) {
  const double tile_idx_key_multiplier = *max_z + 1.0f;
  int splat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (splat_idx >= num_splats)
    return;

  int current_tile_id = floor(sorted_keys[splat_idx] / tile_idx_key_multiplier);

  if (splat_idx == 0) {
    splat_start_end_idx_by_tile_idx[current_tile_id] = 0;
  } else {
    int prev_tile_id = floor(sorted_keys[splat_idx - 1] / tile_idx_key_multiplier);
    if (current_tile_id != prev_tile_id) {
      splat_start_end_idx_by_tile_idx[current_tile_id] = splat_idx;
      splat_start_end_idx_by_tile_idx[prev_tile_id + 1] = splat_idx;
    }
  }

  if (splat_idx == num_splats - 1) {
    splat_start_end_idx_by_tile_idx[current_tile_id + 1] = num_splats;
  }
}

void cull_gaussians(float *const uv, float *const xyz, const int N, const float near_thresh, const float far_thresh,
                    const int padding, const int width, const int height, bool *mask, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(uv);
  ASSERT_DEVICE_POINTER(xyz);
  ASSERT_DEVICE_POINTER(mask);

  const int threads_per_block = 256;
  // Calculate the number of blocks needed to cover all N points
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  frustum_culling_kernel<<<gridsize, blocksize, 0, stream>>>(uv, xyz, N, near_thresh, far_thresh, padding, width,
                                                             height, mask);
}

__global__ void scatter_add_gradients_kernel(const float *d_grad_uv_culled, const float *d_grad_xyz_culled,
                                             const bool *d_mask, const int *d_culled_count_prefix_sum,
                                             int num_gaussians, float *d_uv_grad_accum_full,
                                             float *d_xyz_grad_accum_full, int *d_grad_accum_dur_full) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_gaussians)
    return;

  if (d_mask[idx]) {
    int culled_idx = d_culled_count_prefix_sum[idx];

    atomicAdd(&d_uv_grad_accum_full[idx * 2 + 0], d_grad_uv_culled[culled_idx * 2 + 0]);
    atomicAdd(&d_uv_grad_accum_full[idx * 2 + 1], d_grad_uv_culled[culled_idx * 2 + 1]);

    atomicAdd(&d_xyz_grad_accum_full[idx * 3 + 0], d_grad_xyz_culled[culled_idx * 3 + 0]);
    atomicAdd(&d_xyz_grad_accum_full[idx * 3 + 1], d_grad_xyz_culled[culled_idx * 3 + 1]);
    atomicAdd(&d_xyz_grad_accum_full[idx * 3 + 2], d_grad_xyz_culled[culled_idx * 3 + 2]);

    atomicAdd(&d_grad_accum_dur_full[idx], 1);
  }
}

__global__ void scatter_groups_kernel(const float *input, const bool *mask, const int *scan_out, const int N,
                                      float *output, int S) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= N) {
    return;
  }
  if (mask[gid]) {
    int in_group = scan_out[gid];
    for (int j = 0; j < S; ++j) {
      output[gid * S + j] = input[in_group * S + j];
    }
  }
}

__global__ void select_groups_kernel(const float *input, const bool *mask, const int *scan_out, const int N,
                                     float *output, int S) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x; // group id
  if (gid >= N)
    return;
  if (mask[gid]) {
    int out_group = scan_out[gid];
    for (int j = 0; j < S; ++j) {
      output[out_group * S + j] = input[gid * S + j];
    }
  }
}

__global__ void getTotalSum(const int *mask_sum, const bool *d_mask, int *d_num_culled, int N) {
  if (N > 0) {
    // The total sum = exclusive_scan_result[N-1] + input[N-1]
    *d_num_culled = mask_sum[N - 1] + static_cast<int>(d_mask[N - 1]);
  } else {
    *d_num_culled = 0;
  }
}

void filter_moment_vectors(const int N, const int S, const bool *d_mask, const float *d_m, const float *d_v,
                           float *d_m_culled, float *d_v_culled, cudaStream_t stream) {
  int *mask_sum;
  CHECK_CUDA(cudaMalloc(&mask_sum, N * sizeof(int)));
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_mask, mask_sum, N);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_mask, mask_sum, N);

  const int threads_per_block = 256;
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  // Apply mask
  select_groups_kernel<<<num_blocks, threads_per_block, 0, stream>>>(d_m, d_mask, mask_sum, N, d_m_culled, S);
  select_groups_kernel<<<num_blocks, threads_per_block, 0, stream>>>(d_v, d_mask, mask_sum, N, d_v_culled, S);

  // Free the temporary storage.
  CHECK_CUDA(cudaFree(d_temp_storage));
  CHECK_CUDA(cudaFree(mask_sum));
}

void accumulate_gradients(const int N, const bool *d_mask, const float *d_grad_xyz, const float *d_grad_uv,
                          float *d_xyz_grad_accum, float *d_uv_grad_acuum, int *d_grad_accum_dur, cudaStream_t stream) {
  int *mask_sum;
  CHECK_CUDA(cudaMalloc(&mask_sum, N * sizeof(int)));
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_mask, mask_sum, N);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_mask, mask_sum, N);

  const int threads_per_block = 256;
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  scatter_add_gradients_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      d_grad_uv, d_grad_xyz, d_mask, mask_sum, N, d_uv_grad_acuum, d_xyz_grad_accum, d_grad_accum_dur);

  // Free the temporary storage.
  CHECK_CUDA(cudaFree(d_temp_storage));
  CHECK_CUDA(cudaFree(mask_sum));
}

void scatter_params(const int N, const int S, const bool *d_mask, const float *selected_params, float *scattered_params,
                    cudaStream_t stream) {
  int *mask_sum;
  CHECK_CUDA(cudaMalloc(&mask_sum, N * sizeof(int)));
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_mask, mask_sum, N);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_mask, mask_sum, N);

  const int threads_per_block = 256;
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  scatter_groups_kernel<<<num_blocks, threads_per_block, 0, stream>>>(selected_params, d_mask, mask_sum, N,
                                                                      scattered_params, S);

  // Free the temporary storage.
  CHECK_CUDA(cudaFree(d_temp_storage));
  CHECK_CUDA(cudaFree(mask_sum));
}

void filter_gaussians_by_mask(const int N, const int num_sh_coef, const bool *d_mask, const float *d_xyz,
                              const float *d_rgb, const float *d_sh, const float *d_opacity, const float *d_scale,
                              const float *d_quaternion, const float *d_uv, const float *d_xyz_c, float *d_xyz_culled,
                              float *d_rgb_culled, float *d_sh_culled, float *d_opacity_culled, float *d_scale_culled,
                              float *d_quaternion_culled, float *d_uv_culled, float *d_xyz_c_culled, int *h_num_culled,
                              cudaStream_t stream) {
  int *mask_sum;
  CHECK_CUDA(cudaMalloc(&mask_sum, N * sizeof(int)));

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_mask, mask_sum, N);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_mask, mask_sum, N);

  // Copy sum to host
  int *d_num_culled_temp;
  CHECK_CUDA(cudaMalloc(&d_num_culled_temp, sizeof(int)));
  getTotalSum<<<1, 1, 0, stream>>>(mask_sum, d_mask, d_num_culled_temp, N);
  CHECK_CUDA(cudaMemcpyAsync(h_num_culled, d_num_culled_temp, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaFree(d_num_culled_temp));

  const int threads_per_block = 256;
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  // Apply mask to all arrays
  select_groups_kernel<<<num_blocks, threads_per_block, 0, stream>>>(d_xyz, d_mask, mask_sum, N, d_xyz_culled, 3);
  select_groups_kernel<<<num_blocks, threads_per_block, 0, stream>>>(d_rgb, d_mask, mask_sum, N, d_rgb_culled, 3);
  select_groups_kernel<<<num_blocks, threads_per_block, 0, stream>>>(d_sh, d_mask, mask_sum, N, d_sh_culled,
                                                                     num_sh_coef * 3);
  select_groups_kernel<<<num_blocks, threads_per_block, 0, stream>>>(d_opacity, d_mask, mask_sum, N, d_opacity_culled,
                                                                     1);
  select_groups_kernel<<<num_blocks, threads_per_block, 0, stream>>>(d_scale, d_mask, mask_sum, N, d_scale_culled, 3);
  select_groups_kernel<<<num_blocks, threads_per_block, 0, stream>>>(d_quaternion, d_mask, mask_sum, N,
                                                                     d_quaternion_culled, 4);
  select_groups_kernel<<<num_blocks, threads_per_block, 0, stream>>>(d_uv, d_mask, mask_sum, N, d_uv_culled, 2);
  select_groups_kernel<<<num_blocks, threads_per_block, 0, stream>>>(d_xyz_c, d_mask, mask_sum, N, d_xyz_c_culled, 3);

  // Free the temporary storage.
  CHECK_CUDA(cudaFree(d_temp_storage));
  CHECK_CUDA(cudaFree(mask_sum));
}

void get_sorted_gaussian_list(const float *uv, const float *xyz, const float *conic, const int n_tiles_x,
                              const int n_tiles_y, const float mh_dist, const int N, size_t &sorted_gaussian_bytes,
                              int *sorted_gaussians, int *splat_start_end_idx_by_tile_idx, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(uv);
  ASSERT_DEVICE_POINTER(xyz);
  ASSERT_DEVICE_POINTER(conic);

  const int threads_per_block = 256;
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  const int num_tiles = n_tiles_x * n_tiles_y;

  int *d_buffer_index;
  CHECK_CUDA(cudaMalloc(&d_buffer_index, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_buffer_index, 0, sizeof(int)));

  int *d_buffer_bytes;
  CHECK_CUDA(cudaMalloc(&d_buffer_bytes, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_buffer_bytes, 0, sizeof(int)));

  // get required size for output array
  if (sorted_gaussians == nullptr) {
    coarse_binning_kernel<<<num_blocks, threads_per_block>>>(uv, conic, mh_dist, n_tiles_x, n_tiles_y, N,
                                                             d_buffer_bytes, nullptr, d_buffer_index);
    CHECK_CUDA(cudaGetLastError());

    // update host bytes counter
    int temp_bytes = 0;
    CHECK_CUDA(cudaMemcpy(&temp_bytes, d_buffer_bytes, sizeof(int), cudaMemcpyDeviceToHost));
    sorted_gaussian_bytes = temp_bytes;

    // free memory before return
    CHECK_CUDA(cudaFree(d_buffer_bytes));
    CHECK_CUDA(cudaFree(d_buffer_index));
    return;
  }
  // store pairs of guassians and tiles
  int2 *d_pairs;
  CHECK_CUDA(cudaMalloc(&d_pairs, sorted_gaussian_bytes * 2));

  coarse_binning_kernel<<<num_blocks, threads_per_block>>>(uv, conic, mh_dist, n_tiles_x, n_tiles_y, N, d_buffer_bytes,
                                                           d_pairs, d_buffer_index);
  CHECK_CUDA(cudaGetLastError());

  // get max z depth for key multiplier
  float *d_max_z;
  CHECK_CUDA(cudaMalloc(&d_max_z, sizeof(float)));

  // Copy z values to new array
  float *z_vals;
  CHECK_CUDA(cudaMalloc(&z_vals, N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy2D(z_vals, 1 * sizeof(float), xyz + 2, 3 * sizeof(float), 1 * sizeof(float), N,
                          cudaMemcpyDeviceToDevice));

  void *max_reduce_buffer = nullptr;
  size_t max_reduce_buffer_bytes = 0;

  cub::DeviceReduce::Max(max_reduce_buffer, max_reduce_buffer_bytes, z_vals, d_max_z, N);
  CHECK_CUDA(cudaMalloc(&max_reduce_buffer, max_reduce_buffer_bytes));
  cub::DeviceReduce::Max(max_reduce_buffer, max_reduce_buffer_bytes, z_vals, d_max_z, N);

  CHECK_CUDA(cudaFree(z_vals));
  CHECK_CUDA(cudaFree(max_reduce_buffer));

  int num_pairs;
  CHECK_CUDA(cudaMemcpy(&num_pairs, d_buffer_index, sizeof(int), cudaMemcpyDeviceToHost));
  double *d_sort_keys;
  CHECK_CUDA(cudaMalloc(&d_sort_keys, num_pairs * sizeof(double)));
  int *d_global_splat_counter;
  CHECK_CUDA(cudaMalloc(&d_global_splat_counter, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_global_splat_counter, 0, sizeof(int)));

  const int num_blocks_pairs = (num_pairs + threads_per_block - 1) / threads_per_block;
  generate_splats_kernel<<<num_blocks_pairs, threads_per_block, 0, stream>>>(
      uv, xyz, conic, d_pairs, mh_dist, num_pairs, n_tiles_x, n_tiles_y, d_max_z, sorted_gaussians, d_sort_keys,
      d_global_splat_counter);

  int num_splats;
  CHECK_CUDA(cudaMemcpy(&num_splats, d_global_splat_counter, sizeof(int), cudaMemcpyDeviceToHost));

  {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // Sort keys and apply the same permutation to the gaussian indices (in d_sorted_gaussians)
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_sort_keys, d_sort_keys, sorted_gaussians,
                                    sorted_gaussians, num_splats, 0, sizeof(double) * 8, stream);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_sort_keys, d_sort_keys, sorted_gaussians,
                                    sorted_gaussians, num_splats, 0, sizeof(double) * 8, stream);
    CHECK_CUDA(cudaFree(d_temp_storage));
  }

  CHECK_CUDA(cudaMemset(splat_start_end_idx_by_tile_idx, 0, (num_tiles + 1) * sizeof(int)));

  const int boundary_blocks = (num_splats + threads_per_block - 1) / threads_per_block;
  find_tile_boundaries_kernel<<<boundary_blocks, threads_per_block, 0, stream>>>(d_sort_keys, num_splats, d_max_z,
                                                                                 splat_start_end_idx_by_tile_idx);

  CHECK_CUDA(cudaFree(d_max_z));
  CHECK_CUDA(cudaFree(d_sort_keys));
  CHECK_CUDA(cudaFree(d_buffer_bytes));
  CHECK_CUDA(cudaFree(d_buffer_index));
  CHECK_CUDA(cudaFree(d_pairs));
  CHECK_CUDA(cudaFree(d_global_splat_counter));
}
