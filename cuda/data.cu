// data.cu

#include "gsplat_cuda/cuda_data.cuh"

#include <exception>
#include <stdlib.h>
#include <thrust/device_vector.h>

GaussianParameters::GaussianParameters(size_t max_gaussians) {
  try {
    d_xyz.resize(max_gaussians * 3);
    d_rgb.resize(max_gaussians * 3);
    d_sh.resize(max_gaussians * 15 * 3);
    d_opacity.resize(max_gaussians);
    d_scale.resize(max_gaussians * 3);
    d_quaternion.resize(max_gaussians * 4);
  } catch (const std::exception &e) {
    fprintf(stderr, "CUDA Memory Allocation Error (GaussianParameters): %s\n", e.what());
    exit(EXIT_FAILURE);
  }
}

OptimizerParameters::OptimizerParameters(size_t max_gaussians) {
  try {
    d_training_steps.resize(max_gaussians);

    m_grad_xyz.resize(max_gaussians * 3);
    m_grad_rgb.resize(max_gaussians * 3);
    m_grad_sh.resize(max_gaussians * 15 * 3);
    m_grad_opacity.resize(max_gaussians);
    m_grad_scale.resize(max_gaussians * 3);
    m_grad_quaternion.resize(max_gaussians * 4);

    v_grad_xyz.resize(max_gaussians * 3);
    v_grad_rgb.resize(max_gaussians * 3);
    v_grad_sh.resize(max_gaussians * 15 * 3);
    v_grad_opacity.resize(max_gaussians);
    v_grad_scale.resize(max_gaussians * 3);
    v_grad_quaternion.resize(max_gaussians * 4);
  } catch (const std::exception &e) {
    fprintf(stderr, "CUDA Memory Allocation Error (OptimizerParameters): %s\n", e.what());
    exit(EXIT_FAILURE);
  }
}

GaussianGradients::GaussianGradients(size_t max_gaussians) {
  try {
    d_grad_xyz.resize(max_gaussians * 3);
    d_grad_rgb.resize(max_gaussians * 3);
    d_grad_sh.resize(max_gaussians * 15 * 3);
    d_grad_opacity.resize(max_gaussians);
    d_grad_scale.resize(max_gaussians * 3);
    d_grad_quaternion.resize(max_gaussians * 4);

    d_grad_conic.resize(max_gaussians * 3);
    d_grad_uv.resize(max_gaussians * 2);
    d_grad_J.resize(max_gaussians * 6);
    d_grad_sigma.resize(max_gaussians * 9);
    d_grad_xyz_c.resize(max_gaussians * 3);
    d_grad_precompute_rgb.resize(max_gaussians * 3);
  } catch (const std::exception &e) {
    fprintf(stderr, "CUDA Memory Allocation Error (GaussianGradients): %s\n", e.what());
    exit(EXIT_FAILURE);
  }
}

GradientAccumulators::GradientAccumulators(size_t max_gaussians) {
  try {
    d_xyz_grad_accum.resize(max_gaussians * 3);
    d_uv_grad_accum.resize(max_gaussians * 2);
    d_grad_accum_dur.resize(max_gaussians);
  } catch (const std::exception &e) {
    fprintf(stderr, "CUDA Memory Allocation Error (GradientAccumulators): %s\n", e.what());
    exit(EXIT_FAILURE);
  }
}

CameraParameters::CameraParameters() {
  try {
    // Allocate camera parameters
    d_K.resize(9);  // 3x3 matrix
    d_T.resize(12); // 3x4 matrix
  } catch (const std::exception &e) {
    fprintf(stderr, "CUDA Memory Allocation Error (CudaDataManager): %s\n", e.what());
    exit(EXIT_FAILURE);
  }
}

CudaDataManager::CudaDataManager(size_t max_gaussians_in)
    : max_gaussians(max_gaussians_in), gaussians(max_gaussians_in), optimizer(max_gaussians_in),
      gradients(max_gaussians_in), accumulators(max_gaussians_in), camera() {}

template <int STRIDE> struct strided_index_functor {
  __host__ __device__ int operator()(int i) const { return i / STRIDE; }
};

template <int STRIDE, typename T, typename MaskType>
thrust::device_vector<T> compact_masked_array(const thrust::device_vector<T> &d_source,
                                              const thrust::device_vector<MaskType> &d_mask, int num_culled) {
  thrust::device_vector<T> d_selected(num_culled * STRIDE);

  if (STRIDE == 1) {
    thrust::copy_if(d_source.begin(), d_source.end(), d_mask.begin(), d_selected.begin());
  } else {
    // Create the strided stencil iterators
    auto count_it = thrust::make_counting_iterator(0);
    auto idx_map = thrust::make_transform_iterator(count_it, strided_index_functor<STRIDE>());
    auto stencil = thrust::make_permutation_iterator(d_mask.begin(), idx_map);

    // Perform the copy using the strided stencil
    thrust::copy_if(d_source.begin(), d_source.end(), stencil, d_selected.begin());
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

    thrust::copy_if(indices, indices + d_destination.size(), d_mask.begin(), d_scatter_map.begin());
    thrust::scatter(d_compacted.begin(), d_compacted.end(), d_scatter_map.begin(), d_destination.begin());

  } else {
    size_t num_culled_elements = d_compacted.size();
    size_t num_blocks = d_mask.size();                       // Total number of blocks in destination
    size_t num_culled_blocks = num_culled_elements / STRIDE; // Num blocks in source

    thrust::device_vector<int> d_scatter_block_indices(num_culled_blocks);
    auto block_indices = thrust::make_counting_iterator(0);

    thrust::copy_if(block_indices, block_indices + num_blocks, d_mask.begin(), d_scatter_block_indices.begin());
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

    thrust::copy_if(indices, indices + d_destination.size(), d_mask.begin(), d_scatter_map.begin());

  } else {
    size_t num_blocks = d_mask.size();
    size_t num_culled_blocks = num_culled_elements / STRIDE;

    thrust::device_vector<int> d_scatter_block_indices(num_culled_blocks);
    auto block_indices = thrust::make_counting_iterator(0);

    thrust::copy_if(block_indices, block_indices + num_blocks, d_mask.begin(), d_scatter_block_indices.begin());
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

  thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_culled_elements), functor);
}
