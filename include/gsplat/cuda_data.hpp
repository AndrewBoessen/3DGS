// cuda_data.hpp

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Macro for checking CUDA API calls for errors.
#define CHECK_CUDA(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));               \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

// Data structure to manage CUDA device memory using RAII.
// Its lifetime spans the entire training process.
struct CudaDataManager {
  const size_t max_gaussians;

  // Gaussian parameters
  float *d_xyz, *d_rgb, *d_opacity, *d_scale, *d_quaternion;

  // Gradients
  float *d_grad_xyz, *d_grad_rgb, *d_grad_opacity, *d_grad_scale, *d_grad_quaternion;

  // Intermediate gradient buffers
  float *d_grad_conic, *d_grad_uv, *d_grad_J, *d_grad_sigma, *d_grad_xyz_c;

  // Camera parameters
  float *d_K, *d_T;

  // Temporary buffers for processing
  float *d_uv, *d_xyz_c;
  bool *d_mask;

  // Arrays for culled values
  float *d_xyz_culled, *d_rgb_culled, *d_opacity_culled, *d_scale_culled, *d_quaternion_culled, *d_uv_culled,
      *d_xyz_c_culled;

  // Optimizer moment vectors
  float *m_grad_xyz, *m_grad_rgb, *m_grad_opacity, *m_grad_scale, *m_grad_quaternion;
  float *v_grad_xyz, *v_grad_rgb, *v_grad_opacity, *v_grad_scale, *v_grad_quaternion;

  // Filtered moment vectors
  float *m_grad_xyz_culled, *m_grad_rgb_culled, *m_grad_opacity_culled, *m_grad_scale_culled, *m_grad_quaternion_culled;
  float *v_grad_xyz_culled, *v_grad_rgb_culled, *v_grad_opacity_culled, *v_grad_scale_culled, *v_grad_quaternion_culled;

  CudaDataManager(size_t max_gaussians_in);
  ~CudaDataManager();
};

// Data structure to hold pointers for buffers allocated per-iteration.
struct ForwardPassData {
  int num_culled = 0;
  // Buffers dependent on num_culled
  float *d_sigma = nullptr, *d_conic = nullptr, *d_J = nullptr;
  // Buffers for sorting
  int *d_sorted_gaussians = nullptr, *d_splat_start_end_idx_by_tile_idx = nullptr;
  // Buffers for rendering
  float *d_image_buffer = nullptr, *d_weight_per_pixel = nullptr;
  int *d_splats_per_pixel = nullptr;
};
