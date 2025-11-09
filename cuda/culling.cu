#include "checks.cuh"
#include "gsplat_cuda/cuda_forward.cuh"
#include <cassert>
#include <cub/cub.cuh>
#include <math_constants.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

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

__device__ __forceinline__ float compute_obb(const float u, const float v, const float a, const float b, const float c,
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

  // Return major axis radius
  return r_major;
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
                                      int *buffer_size, int2 *pairs, int *global_index, float *radii) {
  const int gaussian_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // mask active threads for warpSum
  unsigned active_mask = __ballot_sync(0xFFFFFFFF, gaussian_idx < N);

  if (gaussian_idx >= N)
    return;

  const float u = uvs[gaussian_idx * 2];
  const float v = uvs[gaussian_idx * 2 + 1];
  const float a = conic[gaussian_idx * 3 + 0] + 0.3f;
  const float b = conic[gaussian_idx * 3 + 1];
  const float c = conic[gaussian_idx * 3 + 2] + 0.3f;

  float obb[8];
  const float radius = compute_obb(u, v, a, b, c, mh_dist, obb);
  const int radius_tiles = ceilf(radius * 0.0625f) + 1;

  const float curr_max_radius = radii[gaussian_idx];
  radii[gaussian_idx] = fmaxf(radius, curr_max_radius);

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
      atomicAdd(buffer_size, warp_sum);
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
                                       const int num_tiles_y, const float max_z, int *gaussian_idx_by_splat_idx,
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
  const float a = conic[gaussian_idx * 3 + 0] + 0.3f;
  const float b = conic[gaussian_idx * 3 + 1];
  const float c = conic[gaussian_idx * 3 + 2] + 0.3f;

  float obb[8];
  const float radius = compute_obb(u, v, a, b, c, mh_dist, obb);
  const int radius_tiles = ceilf(radius * 0.0625f) + 1;

  const int tile_x = tile_idx % num_tiles_x;
  const int tile_y = tile_idx / num_tiles_x;

  double tile_idx_key_multiplier = 0.0;
  tile_idx_key_multiplier = max_z + 1.0f;

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

__global__ void find_tile_boundaries_kernel(const double *__restrict__ sorted_keys, const int num_splats,
                                            const int num_tiles, const float max_z,
                                            int *__restrict__ splat_start_end_idx_by_tile_idx) {
  const double tile_idx_key_multiplier = (double)max_z + 1.0f;

  int splat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (splat_idx >= num_splats) {
    return;
  }

  int current_tile_id = floor(sorted_keys[splat_idx] / tile_idx_key_multiplier);
  current_tile_id = min(max(current_tile_id, 0), num_tiles - 1);

  int prev_tile_id;
  const bool is_first_splat = (splat_idx == 0);

  if (is_first_splat) {
    // The first splat is always a boundary. We can think of the "previous"
    // tile as -1 to ensure the boundary logic triggers correctly.
    prev_tile_id = -1;
  } else {
    prev_tile_id = floor(sorted_keys[splat_idx - 1] / tile_idx_key_multiplier);
    prev_tile_id = min(max(prev_tile_id, 0), num_tiles - 1);
  }

  // If this thread's splat has a different tile ID than the previous one,
  // it's responsible for writing the new boundary location.
  if (current_tile_id > prev_tile_id) {
    for (int i = prev_tile_id + 1; i <= current_tile_id; ++i) {
      splat_start_end_idx_by_tile_idx[i] = splat_idx;
    }
  }

  // The thread processing the very last splat is responsible for writing the final
  // boundary marker. This marks the end of the last non-empty tile and fills
  // in the markers for any subsequent empty tiles.
  if (splat_idx == num_splats - 1) {
    for (int i = current_tile_id + 1; i <= num_tiles; ++i) {
      splat_start_end_idx_by_tile_idx[i] = num_splats;
    }
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

// Helper functor for strided copy (xyz -> z)
struct copy_z_functor {
  const float *m_xyz;
  copy_z_functor(const float *xyz) : m_xyz(xyz) {}
  __host__ __device__ float operator()(int i) const {
    return m_xyz[i * 3 + 2]; // Get the z-component
  }
};

void get_sorted_gaussian_list(const float *uv, const float *xyz, const float *conic, const int n_tiles_x,
                              const int n_tiles_y, const float mh_dist, const int N, size_t &sorted_gaussian_size,
                              int *sorted_gaussians, int *splat_start_end_idx_by_tile_idx, float *radii,
                              cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(uv);
  ASSERT_DEVICE_POINTER(xyz);
  ASSERT_DEVICE_POINTER(conic);

  const int threads_per_block = 256;
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  const int num_tiles = n_tiles_x * n_tiles_y;

  // get required size for output array
  if (sorted_gaussians == nullptr) {
    // Use device_vectors for atomic counters. Initialize to 0.
    thrust::device_vector<int> d_buffer_size(1, 0);

    coarse_binning_kernel<<<num_blocks, threads_per_block>>>(uv, conic, mh_dist, n_tiles_x, n_tiles_y, N,
                                                             thrust::raw_pointer_cast(d_buffer_size.data()), nullptr,
                                                             nullptr, radii);
    sorted_gaussian_size = d_buffer_size[0];

    return;
  }

  // --- Main execution path ---

  // Use device_vectors for atomic counters, initialized to 0
  thrust::device_vector<int> d_buffer_index(1, 0);

  // store pairs of gaussians and tiles
  thrust::device_vector<int2> d_pairs(sorted_gaussian_size);

  coarse_binning_kernel<<<num_blocks, threads_per_block>>>(uv, conic, mh_dist, n_tiles_x, n_tiles_y, N, nullptr,
                                                           thrust::raw_pointer_cast(d_pairs.data()),
                                                           thrust::raw_pointer_cast(d_buffer_index.data()), radii);
  assert(d_buffer_index[0] == sorted_gaussian_size);

  // Copy z values to new array
  thrust::device_vector<float> z_vals(N);

  // Use thrust::transform for the strided copy (replaces cudaMemcpy2D)
  // This executes on the device using the default stream, matching original async behavior
  thrust::transform(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(N),
                    z_vals.begin(), copy_z_functor(xyz));

  // Use thrust::max_element to find max Z value (replaces CUB::DeviceReduce)
  // This is a device-side operation
  auto max_iter = thrust::max_element(thrust::device, z_vals.begin(), z_vals.end());
  // Copy the single max value to d_max_z
  const float max_z = max_iter[0];

  // Get num_pairs from device
  int num_pairs = d_buffer_index[0]; // Device-to-host copy

  thrust::device_vector<double> d_sort_keys(num_pairs);
  thrust::device_vector<int> d_global_splat_counter(1, 0); // Initialize to 0

  const int num_blocks_pairs = (num_pairs + threads_per_block - 1) / threads_per_block;
  generate_splats_kernel<<<num_blocks_pairs, threads_per_block, 0, stream>>>(
      uv, xyz, conic, thrust::raw_pointer_cast(d_pairs.data()), mh_dist, num_pairs, n_tiles_x, n_tiles_y, max_z,
      sorted_gaussians, // Pass through the raw pointer from caller
      thrust::raw_pointer_cast(d_sort_keys.data()), thrust::raw_pointer_cast(d_global_splat_counter.data()));

  int num_splats = d_global_splat_counter[0]; // Device-to-host copy

  {
    // Use device_vector for CUB temporary storage
    size_t temp_storage_bytes = 0;

    // First call to get storage size
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                    thrust::raw_pointer_cast(d_sort_keys.data()), // keys_in
                                    thrust::raw_pointer_cast(d_sort_keys.data()), // keys_out
                                    sorted_gaussians,                             // values_in
                                    sorted_gaussians,                             // values_out
                                    num_splats, 0, sizeof(double) * 8, stream);

    // Allocate temp storage using device_vector
    thrust::device_vector<char> d_temp_storage(temp_storage_bytes);

    // Second call to perform sort
    cub::DeviceRadixSort::SortPairs(thrust::raw_pointer_cast(d_temp_storage.data()), // Pass raw pointer
                                    temp_storage_bytes, thrust::raw_pointer_cast(d_sort_keys.data()),
                                    thrust::raw_pointer_cast(d_sort_keys.data()), sorted_gaussians, sorted_gaussians,
                                    num_splats, 0, sizeof(double) * 8, stream);
  }

  const int boundary_blocks = (num_splats + threads_per_block - 1) / threads_per_block;
  find_tile_boundaries_kernel<<<boundary_blocks, threads_per_block, 0, stream>>>(
      thrust::raw_pointer_cast(d_sort_keys.data()), num_splats, num_tiles, max_z, splat_start_end_idx_by_tile_idx);
}
