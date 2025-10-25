// cuda_data.cuh

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <thrust/device_vector.h>

// Holds the core parameters for each Gaussian splat.
struct GaussianParameters {
  // Gaussian parameters
  thrust::device_vector<float> d_xyz, d_rgb, d_sh, d_opacity, d_scale, d_quaternion;

  GaussianParameters(size_t max_gaussians);
};

// Holds the momentum vectors for the Adam optimizer.
struct OptimizerParameters {
  // A device-side variable to track training steps, can be used for learning rate schedules.
  thrust::device_vector<int> d_training_steps;

  // Optimizer moment vectors
  thrust::device_vector<float> m_grad_xyz, m_grad_rgb, m_grad_sh, m_grad_opacity, m_grad_scale, m_grad_quaternion;
  thrust::device_vector<float> v_grad_xyz, v_grad_rgb, v_grad_sh, v_grad_opacity, v_grad_scale, v_grad_quaternion;

  OptimizerParameters(size_t max_gaussians);
};

// Holds the gradients for the Gaussian parameters.
struct GaussianGradients {
  // Gradients corresponding to GaussianParameters
  thrust::device_vector<float> d_grad_xyz, d_grad_rgb, d_grad_sh, d_grad_opacity, d_grad_scale, d_grad_quaternion;

  // Intermediate gradient buffers used during backpropagation
  thrust::device_vector<float> d_grad_conic, d_grad_uv, d_grad_J, d_grad_sigma, d_grad_xyz_c, d_grad_precompute_rgb;

  GaussianGradients(size_t max_gaussians);
};

// Holds buffers for accumulating gradients, used for density control heuristics.
struct GradientAccumulators {
  // Gradient accumulators
  thrust::device_vector<float> d_xyz_grad_accum, d_uv_grad_accum;
  thrust::device_vector<int> d_grad_accum_dur;

  GradientAccumulators(size_t max_gaussians);
};

// Holds buffer to storing current camera parameters
struct CameraParameters {
  // Camera parameters
  thrust::device_vector<float> d_K, d_T;

  CameraParameters();
};

// Top-level data structure that owns and manages all persistent CUDA device memory.
// Its lifetime should span the entire training process.
struct CudaDataManager {
  const size_t max_gaussians;

  GaussianParameters gaussians;
  OptimizerParameters optimizer;
  GaussianGradients gradients;
  GradientAccumulators accumulators;
  CameraParameters camera;

  CudaDataManager(size_t max_gaussians_in);
};

// Data structure to hold pointers for buffers allocated temporarily for each forward pass.
struct ForwardPassData {
  size_t num_culled = 0;

  // Buffers dependent on num_culled
  thrust::device_vector<float> d_sigma, d_conic, d_J, d_precomputed_rgb;

  // Temporary buffers for processing
  thrust::device_vector<float> d_uv, d_xyz_c;
  thrust::device_vector<bool> d_mask;

  // Buffers for sorting
  thrust::device_vector<int> d_sorted_gaussians, d_splat_start_end_idx_by_tile_idx;
  // Buffers for rendering
  thrust::device_vector<float> d_image_buffer, d_weight_per_pixel;
  thrust::device_vector<int> d_splats_per_pixel;
};

struct is_true {
  __host__ __device__ bool operator()(const bool x) { return x; }
};

template <int STRIDE> struct strided_index_functor {
  __host__ __device__ int operator()(int i) const { return i / STRIDE; }
};

/**
 * @brief Compacts a strided device_vector based on a mask.
 * @tparam STRIDE The number of elements per item in the source vector (e.g., 3 for XYZ).
 * @tparam T The data type of the source/destination vector (e.g., float).
 * @tparam MaskType The data type of the mask vector (e.g., int, bool).
 * @param d_source The original, large, strided source vector.
 * @param d_mask The mask vector (size = d_source.size() / STRIDE).
 * @param num_culled The number of 'true' values in the mask (pre-calculated).
 * @return A new, compacted device_vector of size (num_culled * STRIDE).
 */
template <int STRIDE, typename T, typename MaskType>
thrust::device_vector<T> compact_masked_array(const thrust::device_vector<T> &d_source,
                                              const thrust::device_vector<MaskType> &d_mask, int num_culled) {
  thrust::device_vector<T> d_selected(num_culled * STRIDE);

  if (STRIDE == 1) {
    thrust::copy_if(d_source.begin(), d_source.begin() + d_mask.size(), d_mask.begin(), d_selected.begin(), is_true());
  } else {
    // Create the strided stencil iterators
    auto count_it = thrust::make_counting_iterator(0);
    auto idx_map = thrust::make_transform_iterator(count_it, strided_index_functor<STRIDE>());
    auto stencil = thrust::make_permutation_iterator(d_mask.begin(), idx_map);

    // Perform the copy using the strided stencil
    thrust::copy_if(d_source.begin(), d_source.begin() + d_mask.size() * STRIDE, stencil, d_selected.begin(),
                    is_true());
  }

  return d_selected;
}

template <int STRIDE> struct scatter_map_functor {
  const int *d_block_indices;

  scatter_map_functor(const int *ptr) : d_block_indices(ptr) {}

  __host__ __device__ int operator()(int i) const {
    int block_index = i / STRIDE;
    int intra_block_offset = i % STRIDE;

    // Get the true destination block index from our compacted list
    int destination_block = d_block_indices[block_index];

    // Calculate the final destination index in the large array
    return destination_block * STRIDE + intra_block_offset;
  }
};

/**
 * @brief Scatters a compacted array back into a larger destination array based on a mask.
 * @tparam STRIDE The stride of the data blocks.
 * @tparam T The element type.
 * @tparam MaskType The mask element type (e.g., bool, int).
 * @param d_compacted The compacted source vector (output from `compact_masked_array`).
 * @param d_mask The original mask used for compaction.
 * @param d_destination The large destination vector to be overwritten.
 */
template <int STRIDE, typename T, typename MaskType>
void scatter_masked_array(const thrust::device_vector<T> &d_compacted, const thrust::device_vector<MaskType> &d_mask,
                          thrust::device_vector<T> &d_destination) {

  if (d_compacted.empty()) {
    return;
  }

  if (STRIDE == 1) {
    size_t num_culled = d_compacted.size();
    thrust::device_vector<int> d_scatter_map(num_culled);

    auto indices = thrust::make_counting_iterator(0);

    thrust::copy_if(indices, indices + d_mask.size(), d_mask.begin(), d_scatter_map.begin(), is_true());
    thrust::scatter(d_compacted.begin(), d_compacted.end(), d_scatter_map.begin(), d_destination.begin());

  } else {
    size_t num_culled_elements = d_compacted.size();
    size_t num_blocks = d_mask.size();                       // Total number of blocks in destination
    size_t num_culled_blocks = num_culled_elements / STRIDE; // Num blocks in source

    thrust::device_vector<int> d_scatter_block_indices(num_culled_blocks);
    auto block_indices = thrust::make_counting_iterator(0);

    thrust::copy_if(block_indices, block_indices + num_blocks, d_mask.begin(), d_scatter_block_indices.begin(),
                    is_true());
    thrust::device_vector<int> d_scatter_map(num_culled_elements);

    auto source_indices = thrust::make_counting_iterator(0);
    auto map_it = thrust::make_transform_iterator(
        source_indices, scatter_map_functor<STRIDE>(thrust::raw_pointer_cast(d_scatter_block_indices.data())));

    thrust::copy(map_it, map_it + num_culled_elements, d_scatter_map.begin());
    thrust::scatter(d_compacted.begin(), d_compacted.end(), d_scatter_map.begin(), d_destination.begin());
  }
}

template <typename T> struct scatter_add_functor {
  const T *d_source;
  const int *d_map;
  T *d_destination;

  scatter_add_functor(const T *src, const int *map, T *dest) : d_source(src), d_map(map), d_destination(dest) {}

  __device__ void operator()(int i) const {
    // Get the value to add
    T value_to_add = d_source[i];

    // Get the destination index from the map
    int dest_index = d_map[i];

    // Perform the atomic add
    // This is a built-in CUDA C++ function
    atomicAdd(&d_destination[dest_index], value_to_add);
  }
};

/**
 * @brief Scatters a compacted array into a larger destination array by
 * *adding* its values to the existing elements.
 * @tparam STRIDE The stride of the data blocks.
 * @tparam T The element type (must be supported by atomicAdd, e.g., int, float, double).
 * @tparam MaskType The mask element type.
 * @param d_compacted The compacted source vector.
 * @param d_mask The original mask used for compaction.
 * @param d_destination The large destination vector to which values will be added.
 */
template <int STRIDE, typename T, typename MaskType>
void scatter_add_masked_array(const thrust::device_vector<T> &d_compacted,
                              const thrust::device_vector<MaskType> &d_mask, thrust::device_vector<T> &d_destination) {

  size_t num_culled_elements = d_compacted.size();
  if (num_culled_elements == 0) {
    return;
  }

  thrust::device_vector<int> d_scatter_map(num_culled_elements);

  if (STRIDE == 1) {
    auto indices = thrust::make_counting_iterator(0);

    thrust::copy_if(indices, indices + d_mask.size(), d_mask.begin(), d_scatter_map.begin(), is_true());

  } else {
    size_t num_blocks = d_mask.size();
    size_t num_culled_blocks = num_culled_elements / STRIDE;

    thrust::device_vector<int> d_scatter_block_indices(num_culled_blocks);
    auto block_indices = thrust::make_counting_iterator(0);

    thrust::copy_if(block_indices, block_indices + num_blocks, d_mask.begin(), d_scatter_block_indices.begin(),
                    is_true());
    auto source_indices = thrust::make_counting_iterator(0);

    auto map_it = thrust::make_transform_iterator(
        source_indices, scatter_map_functor<STRIDE>(thrust::raw_pointer_cast(d_scatter_block_indices.data())));
    thrust::copy(map_it, map_it + num_culled_elements, d_scatter_map.begin());
  }

  // --- Perform the Scatter-Add Operation ---

  const T *p_compacted = thrust::raw_pointer_cast(d_compacted.data());
  const int *p_map = thrust::raw_pointer_cast(d_scatter_map.data());
  T *p_dest = thrust::raw_pointer_cast(d_destination.data());

  scatter_add_functor<T> functor(p_compacted, p_map, p_dest);

  thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(num_culled_elements),
                   functor);
}
