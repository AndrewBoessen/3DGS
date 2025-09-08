#include "checks.cuh"
#include "gsplat/cuda_forward.hpp"
#include <cub/cub.cuh>
#include <math_constants.h>

__device__ __forceinline__ bool z_distance_culling(const float z, const float near_thresh, const float far_thresh) {
  return z >= near_thresh && z <= far_thresh;
}
__device__ __forceinline__ bool frustum_culling(const float u, const float v, const int padding, const int width,
                                                const int height) {
  return u >= (-1 * padding) && u <= width && v >= (-1 * padding) && v <= height;
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

  mask[i] = !(z_distance_culling(z, near_thresh, far_thresh) || frustum_culling(u, v, padding, width, height));
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
  const float left = (a + c) / 2.f;
  const float right = sqrtf((a - c) * (a - c) / 4.0f + b * b);
  const float lambda1 = left + right;
  const float lambda2 = left - right;
  const float r_major = mh_dist * sqrtf(lambda1);
  const float r_minor = mh_dist * sqrtf(lambda2);
  float theta = (fabsf(b) < 1e-16) ? (a >= c ? 0.0f : M_PI / 2.f) : atan2f(lambda1 - a, b);
  const float cos_theta = cosf(theta);
  const float sin_theta = sinf(theta);
  obb[0] = -r_major * cos_theta + r_minor * sin_theta + u;
  obb[1] = -r_major * sin_theta - r_minor * cos_theta + v;
  obb[2] = r_major * cos_theta + r_minor * sin_theta + u;
  obb[3] = r_major * sin_theta - r_minor * cos_theta + v;
  obb[4] = -r_major * cos_theta - r_minor * sin_theta + u;
  obb[5] = -r_major * sin_theta + r_minor * cos_theta + v;
  obb[6] = r_major * cos_theta - r_minor * sin_theta + u;
  obb[7] = r_major * sin_theta + r_minor * cos_theta + v;
  return ceilf(r_major / 16.0f) + 1;
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

__device__ __forceinline__ int warpReduceMin(unsigned mask, int val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = min(val, __shfl_down_sync(mask, val, offset));
  }
  // Broadcast the final result from lane 0 to all threads
  return __shfl_sync(mask, val, 0);
}

__device__ __forceinline__ int warpReduceMax(unsigned mask, int val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(mask, val, offset));
  }
  // Broadcast the final result from lane 0 to all threads
  return __shfl_sync(mask, val, 0);
}

__global__ void generate_splats_kernel(const float *__restrict__ uvs, const float *__restrict__ xyz_camera_frame,
                                       const float *__restrict__ conic, const int n_tiles_x, const int n_tiles_y,
                                       const float mh_dist, const int N, float *max_z, int *gaussian_idx_by_splat_idx,
                                       double *sort_keys, int *global_splat_counter) {
  const bool store_values = !(gaussian_idx_by_splat_idx == nullptr || sort_keys == nullptr);

  int gaussian_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Mask of all active threads
  unsigned active_mask = __ballot_sync(0xFFFFFFFF, gaussian_idx < N);

  if (gaussian_idx >= N)
    return;

  const float u = uvs[gaussian_idx * 2];
  const float v = uvs[gaussian_idx * 2 + 1];
  const double z = (double)(xyz_camera_frame[gaussian_idx * 3 + 2]);
  const float a = conic[gaussian_idx * 3] + 0.25f;
  const float b = conic[gaussian_idx * 3 + 1] / 2.0f;
  const float c = conic[gaussian_idx * 3 + 2] + 0.25f;

  float obb[8];
  const int radius_tiles = compute_obb(u, v, a, b, c, mh_dist, obb);

  const int projected_tile_x = floorf(u / 16.0f);
  const int start_tile_x = max(0, projected_tile_x - radius_tiles);
  const int end_tile_x = min(n_tiles_x, projected_tile_x + radius_tiles + 1);
  const int projected_tile_y = floorf(v / 16.0f);
  const int start_tile_y = max(0, projected_tile_y - radius_tiles);
  const int end_tile_y = min(n_tiles_y, projected_tile_y + radius_tiles + 1);

  double tile_idx_key_multiplier = 0.0;
  if (store_values) {
    tile_idx_key_multiplier = *max_z + 1.0f;
  }

  // Reduce to find the minimum start_x and maximum end_x in the warp
  int warp_start_x = warpReduceMin(active_mask, start_tile_x);
  int warp_end_x = warpReduceMax(active_mask, end_tile_x);
  int warp_start_y = warpReduceMin(active_mask, start_tile_y);
  int warp_end_y = warpReduceMax(active_mask, end_tile_y);

  for (int tile_x = warp_start_x; tile_x < warp_end_x; tile_x++) {
    for (int tile_y = warp_start_y; tile_y < warp_end_y; tile_y++) {
      const bool in_thread_bounds =
          (tile_x >= start_tile_x && tile_x < end_tile_x && tile_y >= start_tile_y && tile_y < end_tile_y);

      float tile_bounds[4];
      tile_bounds[0] = __int2float_rn(tile_x) * 16.0f;
      tile_bounds[1] = __int2float_rn(tile_x + 1) * 16.0f;
      tile_bounds[2] = __int2float_rn(tile_y) * 16.0f;
      tile_bounds[3] = __int2float_rn(tile_y + 1) * 16.0f;

      const bool intersects = in_thread_bounds && split_axis_test(obb, tile_bounds);
      const int lane_id = gaussian_idx & 0x1f;

      // get position of splat in global array
      int splat_idx = get_write_index(intersects, lane_id, active_mask, global_splat_counter);

      if (store_values && intersects) {
        gaussian_idx_by_splat_idx[splat_idx] = gaussian_idx;
        const int tile_idx = tile_y * n_tiles_x + tile_x;
        sort_keys[splat_idx] = z + tile_idx_key_multiplier * __int2double_rn(tile_idx);
      }
    }
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

__device__ __forceinline__ float warp_reduce_max(float val) {
  // Iteratively exchange values with threads at a decreasing offset.
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__global__ void max_reduce_strided_kernel(const float *input, float *output, int n, int stride) {
  // Shared memory to store the max value from each warp in the block.
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // 1. Each thread loads its value from global memory.
  float my_val = (i < n) ? input[i * stride] : -CUDART_INF_F;

  // 2. Perform reduction at the warp level.
  float warp_max = warp_reduce_max(my_val);

  // 3. Lane 0 of each warp writes its warp's max to shared memory.
  if ((tid % 32) == 0) {
    sdata[tid / 32] = warp_max;
  }
  __syncthreads();

  // 4. The first warp reduces the results from shared memory.
  // Only threads that will participate in the final reduction load a value.
  float final_val = (tid < blockDim.x / 32) ? sdata[tid] : -CUDART_INF_F;

  float block_max;
  if (tid < 32) {
    // block max will now be in first thread
    block_max = warp_reduce_max(final_val);
  }
  // store block max to output
  if (tid == 0) {
    output[blockIdx.x] = block_max;
  }
}

void cull_gaussians(float *const uv, float *const xyz, const int N, const float near_thresh, const float far_thresh,
                    const int padding, const int width, const int height, bool *mask) {
  ASSERT_DEVICE_POINTER(uv);
  ASSERT_DEVICE_POINTER(xyz);
  ASSERT_DEVICE_POINTER(mask);

  const int threads_per_block = 1024;
  // Calculate the number of blocks needed to cover all N points
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  dim3 gridsize(num_blocks, 1, 1);
  dim3 blocksize(threads_per_block, 1, 1);

  frustum_culling_kernel<<<gridsize, blocksize>>>(uv, xyz, N, near_thresh, far_thresh, padding, width, height, mask);
}

void filter_gaussians_by_mask(int N, const bool *d_mask, const float *d_xyz, const float *d_rgb, const float *d_opacity,
                              const float *d_scale, const float *d_quaternion, const float *d_uv, const float *d_xyz_c,
                              float *d_xyz_culled, float *d_rgb_culled, float *d_opacity_culled, float *d_scale_culled,
                              float *d_quaternion_culled, float *d_uv_culled, float *d_xyz_c_culled,
                              int *d_num_culled) {
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // First, determine the temporary storage size required by CUB.
  // This is a "dry run" that doesn't actually perform the selection.
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_xyz, d_mask, d_xyz_culled, d_num_culled, N);

  // Allocate the temporary storage buffer on the device.
  CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Now, perform the selection for each attribute array, reusing the temp storage.
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_xyz, d_mask, d_xyz_culled, d_num_culled, N);
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_rgb, d_mask, d_rgb_culled, d_num_culled, N);
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_opacity, d_mask, d_opacity_culled, d_num_culled, N);
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_scale, d_mask, d_scale_culled, d_num_culled, N);
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_quaternion, d_mask, d_quaternion_culled,
                             d_num_culled, N);
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_uv, d_mask, d_uv_culled, d_num_culled, N);
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_xyz_c, d_mask, d_xyz_c_culled, d_num_culled, N);

  // Free the temporary storage.
  CHECK_CUDA(cudaFree(d_temp_storage));
}

void get_sorted_gaussian_list(const float *uv, const float *xyz, const float *conic, const int n_tiles_x,
                              const int n_tiles_y, const float mh_dist, const int N, size_t &sorted_gaussian_bytes,
                              int *sorted_gaussians, int *splat_start_end_idx_by_tile_idx) {
  ASSERT_DEVICE_POINTER(uv);
  ASSERT_DEVICE_POINTER(xyz);
  ASSERT_DEVICE_POINTER(conic);

  const int num_tiles = n_tiles_x * n_tiles_y;

  int *d_global_splat_counter;
  CHECK_CUDA(cudaMalloc(&d_global_splat_counter, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_global_splat_counter, 0, sizeof(int)));

  // get required size for output array
  if (sorted_gaussians == nullptr) {
    const int threads_per_block = 256;
    const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    generate_splats_kernel<<<num_blocks, threads_per_block>>>(uv, xyz, conic, n_tiles_x, n_tiles_y, mh_dist, N, nullptr,
                                                              sorted_gaussians, nullptr, d_global_splat_counter);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(&sorted_gaussian_bytes, d_global_splat_counter, sizeof(int), cudaMemcpyDeviceToHost));
    sorted_gaussian_bytes *= sizeof(int);

    CHECK_CUDA(cudaFree(d_global_splat_counter));

    return;
  }

  const int num_splats = sorted_gaussian_bytes / sizeof(int);

  // get max z depth for key multiplier
  float *d_max_z;
  CHECK_CUDA(cudaMalloc(&d_max_z, sizeof(float)));
  {
    const int threads_per_block_reduce = 256;
    // Shared memory needed is one float per warp.
    const int shared_mem_size = (threads_per_block_reduce / 32) * sizeof(float);

    int num_blocks_pass1 = (N + threads_per_block_reduce - 1) / threads_per_block_reduce;
    float *d_partial_max;
    CHECK_CUDA(cudaMalloc(&d_partial_max, num_blocks_pass1 * sizeof(float)));

    max_reduce_strided_kernel<<<num_blocks_pass1, threads_per_block_reduce, shared_mem_size>>>(xyz + 2, d_partial_max,
                                                                                               N, 3);
    CHECK_CUDA(cudaGetLastError());
    // Pass 2: Reduce partial results to a single value
    max_reduce_strided_kernel<<<1, threads_per_block_reduce, shared_mem_size>>>(d_partial_max, d_max_z,
                                                                                num_blocks_pass1, 1);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_partial_max));
  }

  double *d_sort_keys;
  CHECK_CUDA(cudaMalloc(&d_sort_keys, sorted_gaussian_bytes));

  const int threads_per_block = 256;
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  generate_splats_kernel<<<num_blocks, threads_per_block>>>(uv, xyz, conic, n_tiles_x, n_tiles_y, mh_dist, N, d_max_z,
                                                            sorted_gaussians, d_sort_keys, d_global_splat_counter);
  CHECK_CUDA(cudaDeviceSynchronize());

  {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // Sort keys and apply the same permutation to the gaussian indices (in d_sorted_gaussians)
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_sort_keys, d_sort_keys, sorted_gaussians,
                                    sorted_gaussians, num_splats);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_sort_keys, d_sort_keys, sorted_gaussians,
                                    sorted_gaussians, num_splats);
    CHECK_CUDA(cudaFree(d_temp_storage));
  }

  CHECK_CUDA(cudaMemset(splat_start_end_idx_by_tile_idx, 0, (num_tiles + 1) * sizeof(int)));

  const int boundary_blocks = (num_splats + threads_per_block - 1) / threads_per_block;
  find_tile_boundaries_kernel<<<boundary_blocks, threads_per_block>>>(d_sort_keys, num_splats, d_max_z,
                                                                      splat_start_end_idx_by_tile_idx);

  CHECK_CUDA(cudaFree(d_max_z));
  CHECK_CUDA(cudaFree(d_sort_keys));
  CHECK_CUDA(cudaFree(d_global_splat_counter));
}
