// adaptive_density.hpp

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief Launch CUDA kernel to perform adaptive density step
 * @param[in]      N                                      Total number of gaussians
 * @param[in]      iter                                   Iteration number in trainer
 * @param[in]      num_sh_coef                            Total number of spherical harmonics coefficients
 * @param[in]      max_gaussians                          Max gaussians
 * @param[in]      use_delete                             Flag to delete non contributing gaussians
 * @param[in]      use_clone                              Flag to enable clone
 * @param[in]      use_split                              Flag to enable split
 * @param[in]      delete_opacity_threshold               Delete below threshold
 * @param[in]      grad_threshold                         UV gradeint threshold
 * @param[in]      scene_extent                           Scene extent
 * @param[in]      percent_dense                          Scaling factor to split or clone
 * @param[in]      num_split_samples                      Number of guassians to create in split
 * @param[in]      split_scale_factor                     Scale split gaussians
 * @param[in]      uv_grad_accum                          Sum of uv gradient values
 * @param[in]      grad_accum_count                       Number of times gradient is added
 * @param[in]      scale                                  Scale parameter vector
 * @param[in,out]  d_mask                                 Mask of size max_gaussians
 * @param[in]      xyz_grad_accum                         Sum of xyz gradient values
 * @param[in,out]  xyz                                    XYZ parameter vector
 * @param[in,out]  rgb                                    RGB parameter vector
 * @param[in,out]  sh                                     Spherical Harmonics parameter vector
 * @param[in,out]  opacity                                Opacity parameter vector
 * @param[in,out]  quaternion                             Quaternion parameter vector
 * @param[in,out]  m_xyz                                  XYZ first moment vector
 * @param[in,out]  v_xyz                                  XYZ second moment vector
 * @param[in,out]  m_rgb                                  RGB first moment vector
 * @param[in,out]  v_rgb                                  RGB second moment vector
 * @param[in,out]  m_op                                   Opacity first moment vector
 * @param[in,out]  v_op                                   Opacity second moment vector
 * @param[in,out]  m_scale                                Scale first moment vector
 * @param[in,out]  v_scale                                Scale second moment vector
 * @param[in,out]  m_quat                                 Quaternion first moment vector
 * @param[in,out]  v_quat                                 Quaternion second moment vector
 * @param[in,out]  m_sh                                   Spherical Harmonics first moment vector
 * @param[in,out]  v_sh                                   Spherical Harmonics second moment vector
 * @param[in]      stream                                 The CUDA stream to execute on
 * @return Total number of gaussians after step
 */
int adaptive_density(const int N, const int iter, const int num_sh_coef, const int max_gaussians, const bool use_delete,
                     const bool use_clone, const bool use_split, const float delete_opacity_threshold,
                     const float grad_threshold, const float scene_extent, const float percent_dense,
                     const int num_split_samples, const float split_scale_factor, const float *uv_grad_accum,
                     const int *grad_accum_count, float *scale, bool *d_mask, const float *xyz_grad_accum, float *xyz,
                     float *rgb, float *sh, float *opacity, float *quaternion, float *m_xyz, float *v_xyz, float *m_rgb,
                     float *v_rgb, float *m_op, float *v_op, float *m_scale, float *v_scale, float *m_quat,
                     float *v_quat, float *m_sh, float *v_sh, cudaStream_t stream = 0);
