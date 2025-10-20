// data.cu

#include "gsplat/cuda_data.cuh"
#include <exception>
#include <stdlib.h>

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
