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
                                              const thrust::device_vector<MaskType> &d_mask, int num_culled);

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
                          thrust::device_vector<T> &d_destination);

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
                              const thrust::device_vector<MaskType> &d_mask, thrust::device_vector<T> &d_destination);
