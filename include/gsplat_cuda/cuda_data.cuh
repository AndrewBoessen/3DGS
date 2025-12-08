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
  thrust::device_vector<float> d_uv_grad_accum;
  thrust::device_vector<int> d_grad_accum_dur;

  GradientAccumulators(size_t max_gaussians);
};

// Holds buffer to storing current camera parameters
struct CameraParameters {
  // Camera parameters
  thrust::device_vector<float> d_view, d_proj;

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
  thrust::device_vector<float4> d_radius;

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

// Helper functor to compute the final scatter destination index
// from the destination *row* and the element's *global index*.
template <int STRIDE> struct scatter_index_functor {
  __host__ __device__ int operator()(const thrust::tuple<int, int> &t) const {
    int dest_row = thrust::get<0>(t); // The destination row index
    int i = thrust::get<1>(t);        // The global index (0 to num_compacted * STRIDE - 1)
    int offset = i % STRIDE;          // The offset within the stride block
    return dest_row * STRIDE + offset;
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

  size_t num_compacted_rows = d_compacted.size() / STRIDE;
  if (num_compacted_rows == 0) {
    return;
  }

  thrust::device_vector<int> d_row_indices(num_compacted_rows);
  auto count_it = thrust::make_counting_iterator(0);
  thrust::copy_if(count_it, count_it + d_mask.size(), d_mask.begin(), d_row_indices.begin(), is_true());

  auto idx_map = thrust::make_transform_iterator(count_it, strided_index_functor<STRIDE>());
  auto strided_rows_it = thrust::make_permutation_iterator(d_row_indices.begin(), idx_map);

  auto zip_it = thrust::make_zip_iterator(thrust::make_tuple(strided_rows_it, count_it));
  auto d_scatter_map_it = thrust::make_transform_iterator(zip_it, scatter_index_functor<STRIDE>());

  thrust::scatter(d_compacted.begin(), d_compacted.end(), d_scatter_map_it, d_destination.begin());
}
