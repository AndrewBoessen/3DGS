// data.cu

#include "gsplat/cuda_data.hpp"

CudaDataManager::CudaDataManager(size_t max_gaussians_in) : max_gaussians(max_gaussians_in) {
  // Allocate Gaussian parameters
  CHECK_CUDA(cudaMalloc(&d_xyz, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_rgb, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_sh, max_gaussians * 15 * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_opacity, max_gaussians * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_scale, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_quaternion, max_gaussians * 4 * sizeof(float)));

  // Allocate gradients
  CHECK_CUDA(cudaMalloc(&d_grad_xyz, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_grad_rgb, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_grad_sh, max_gaussians * 15 * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_grad_opacity, max_gaussians * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_grad_scale, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_grad_quaternion, max_gaussians * 4 * sizeof(float)));

  // Intermediate gradient buffers
  CHECK_CUDA(cudaMalloc(&d_grad_conic, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_grad_uv, max_gaussians * 2 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_grad_J, max_gaussians * 6 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_grad_sigma, max_gaussians * 9 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_grad_xyz_c, max_gaussians * 3 * sizeof(float)));

  // Camera parameters
  CHECK_CUDA(cudaMalloc(&d_K, 9 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_T, 12 * sizeof(float)));

  // Temporary processing buffers
  CHECK_CUDA(cudaMalloc(&d_uv, max_gaussians * 2 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_xyz_c, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_mask, max_gaussians * sizeof(bool)));

  // Culled value arrays
  CHECK_CUDA(cudaMalloc(&d_xyz_culled, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_rgb_culled, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_sh_culled, max_gaussians * 3 * 15 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_opacity_culled, max_gaussians * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_scale_culled, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_quaternion_culled, max_gaussians * 4 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_uv_culled, max_gaussians * 2 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_xyz_c_culled, max_gaussians * 3 * sizeof(float)));

  // Optimizer moment vectors
  CHECK_CUDA(cudaMalloc(&m_grad_xyz, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&m_grad_rgb, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&m_grad_sh, max_gaussians * 15 * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&m_grad_opacity, max_gaussians * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&m_grad_scale, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&m_grad_quaternion, max_gaussians * 4 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&v_grad_xyz, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&v_grad_rgb, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&v_grad_sh, max_gaussians * 15 * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&v_grad_opacity, max_gaussians * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&v_grad_scale, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&v_grad_quaternion, max_gaussians * 4 * sizeof(float)));

  // Optimizer culled moment vectors
  CHECK_CUDA(cudaMalloc(&m_grad_xyz_culled, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&m_grad_rgb_culled, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&m_grad_sh_culled, max_gaussians * 15 * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&m_grad_opacity_culled, max_gaussians * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&m_grad_scale_culled, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&m_grad_quaternion_culled, max_gaussians * 4 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&v_grad_xyz_culled, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&v_grad_rgb_culled, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&v_grad_sh_culled, max_gaussians * 15 * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&v_grad_opacity_culled, max_gaussians * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&v_grad_scale_culled, max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&v_grad_quaternion_culled, max_gaussians * 4 * sizeof(float)));
}

CudaDataManager::~CudaDataManager() {
  // Free all allocated memory
  CHECK_CUDA(cudaFree(d_xyz));
  CHECK_CUDA(cudaFree(d_rgb));
  CHECK_CUDA(cudaFree(d_sh));
  CHECK_CUDA(cudaFree(d_opacity));
  CHECK_CUDA(cudaFree(d_scale));
  CHECK_CUDA(cudaFree(d_quaternion));
  CHECK_CUDA(cudaFree(d_grad_xyz));
  CHECK_CUDA(cudaFree(d_grad_rgb));
  CHECK_CUDA(cudaFree(d_grad_sh));
  CHECK_CUDA(cudaFree(d_grad_opacity));
  CHECK_CUDA(cudaFree(d_grad_scale));
  CHECK_CUDA(cudaFree(d_grad_quaternion));
  CHECK_CUDA(cudaFree(d_grad_conic));
  CHECK_CUDA(cudaFree(d_grad_uv));
  CHECK_CUDA(cudaFree(d_grad_J));
  CHECK_CUDA(cudaFree(d_grad_sigma));
  CHECK_CUDA(cudaFree(d_grad_xyz_c));
  CHECK_CUDA(cudaFree(d_K));
  CHECK_CUDA(cudaFree(d_T));
  CHECK_CUDA(cudaFree(d_uv));
  CHECK_CUDA(cudaFree(d_xyz_c));
  CHECK_CUDA(cudaFree(d_mask));
  CHECK_CUDA(cudaFree(d_xyz_culled));
  CHECK_CUDA(cudaFree(d_rgb_culled));
  CHECK_CUDA(cudaFree(d_sh_culled));
  CHECK_CUDA(cudaFree(d_opacity_culled));
  CHECK_CUDA(cudaFree(d_scale_culled));
  CHECK_CUDA(cudaFree(d_quaternion_culled));
  CHECK_CUDA(cudaFree(d_uv_culled));
  CHECK_CUDA(cudaFree(d_xyz_c_culled));
  CHECK_CUDA(cudaFree(m_grad_xyz));
  CHECK_CUDA(cudaFree(m_grad_rgb));
  CHECK_CUDA(cudaFree(m_grad_sh));
  CHECK_CUDA(cudaFree(m_grad_opacity));
  CHECK_CUDA(cudaFree(m_grad_scale));
  CHECK_CUDA(cudaFree(m_grad_quaternion));
  CHECK_CUDA(cudaFree(v_grad_xyz));
  CHECK_CUDA(cudaFree(v_grad_rgb));
  CHECK_CUDA(cudaFree(v_grad_sh));
  CHECK_CUDA(cudaFree(v_grad_opacity));
  CHECK_CUDA(cudaFree(v_grad_scale));
  CHECK_CUDA(cudaFree(v_grad_quaternion));
  CHECK_CUDA(cudaFree(m_grad_xyz_culled));
  CHECK_CUDA(cudaFree(m_grad_rgb_culled));
  CHECK_CUDA(cudaFree(m_grad_sh_culled));
  CHECK_CUDA(cudaFree(m_grad_opacity_culled));
  CHECK_CUDA(cudaFree(m_grad_scale_culled));
  CHECK_CUDA(cudaFree(m_grad_quaternion_culled));
  CHECK_CUDA(cudaFree(v_grad_xyz_culled));
  CHECK_CUDA(cudaFree(v_grad_rgb_culled));
  CHECK_CUDA(cudaFree(v_grad_sh_culled));
  CHECK_CUDA(cudaFree(v_grad_opacity_culled));
  CHECK_CUDA(cudaFree(v_grad_scale_culled));
  CHECK_CUDA(cudaFree(v_grad_quaternion_culled));
}
