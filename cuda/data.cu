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

    // Intialize states
    thrust::fill(m_grad_xyz.begin(), m_grad_xyz.end(), 0.0f);
    thrust::fill(m_grad_rgb.begin(), m_grad_rgb.end(), 0.0f);
    thrust::fill(m_grad_sh.begin(), m_grad_sh.end(), 0.0f);
    thrust::fill(m_grad_opacity.begin(), m_grad_opacity.end(), 0.0f);
    thrust::fill(m_grad_scale.begin(), m_grad_scale.end(), 0.0f);
    thrust::fill(m_grad_quaternion.begin(), m_grad_quaternion.end(), 0.0f);

    thrust::fill(v_grad_xyz.begin(), v_grad_xyz.end(), 0.0f);
    thrust::fill(v_grad_rgb.begin(), v_grad_rgb.end(), 0.0f);
    thrust::fill(v_grad_sh.begin(), v_grad_sh.end(), 0.0f);
    thrust::fill(v_grad_opacity.begin(), v_grad_opacity.end(), 0.0f);
    thrust::fill(v_grad_scale.begin(), v_grad_scale.end(), 0.0f);
    thrust::fill(v_grad_quaternion.begin(), v_grad_quaternion.end(), 0.0f);
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
    d_uv_grad_accum.resize(max_gaussians);
    d_max_radii.resize(max_gaussians);
    d_grad_accum_dur.resize(max_gaussians);

    // Zero out accumulators
    thrust::fill(d_xyz_grad_accum.begin(), d_xyz_grad_accum.end(), 0.0f);
    thrust::fill(d_uv_grad_accum.begin(), d_uv_grad_accum.end(), 0.0f);
    thrust::fill(d_max_radii.begin(), d_max_radii.end(), 0.0f);
    thrust::fill(d_grad_accum_dur.begin(), d_grad_accum_dur.end(), 0);
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
