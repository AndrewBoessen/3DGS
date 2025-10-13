// adaptive_density.hpp

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief Launch CUDA kernel to perform adaptive density step
 * @param[in]      N                                      Total number of gaussians
 * @param[in]      iter                                   Iteration number in trainer
 * @param[in]      num_sh_coef                            Total number of spherical harmonics coefficients
 * @param[in]      use_adaptive_fractional_densification  Flag to enable adaptive fractional density control
 * @param[in]      adaptive_control_end                   Last interation number to run adaptive density
 * @param[in]      adaptive_control_start                 First iteration to run adaptive density
 * @param[in]      uv_grad_threshold                      UV gradeint threshold
 * @param[in]      use_fractional_densification           Flag to enable fractional density control
 * @param[in]      uv_grad_percentile                     Quantile percent to split gaussians by
 * @param[in]      scale_norm_percentile                  Scale norm percentile to split
 * @param[in]      max_gaussians                          Max gaussians
 * @param[in]      use_delete                             Flag to delete non contributing gaussians
 * @param[in]      use_clone                              Flag to enable clone
 * @param[in]      use_split                              Flag to enable split
 * @param[in]      delete_opacity_threshold               Delete below threshold
 * @param[in]      clone_scale_threshold                  Clone below threshold
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
 * @param[in]      stream                                 The CUDA stream to execute on
 * @return Total number of gaussians after
 */
int adaptive_density(const int N, const int iter, const int num_sh_coef,
                     const bool use_adaptive_fractional_densification, const int adaptive_control_end,
                     const int adaptive_control_start, const float uv_grad_threshold,
                     const bool use_fractional_densification, const float uv_grad_percentile,
                     const float scale_norm_percentile, const int max_gaussians, const bool use_delete,
                     const bool use_clone, const bool use_split, const float delete_opacity_threshold,
                     const float clone_scale_threshold, const int num_split_samples, const float split_scale_factor,
                     const float *uv_grad_accum, const int *grad_accum_count, float *scale, bool *d_mask,
                     const float *xyz_grad_accum, float *xyz, float *rgb, float *sh, float *opacity, float *quaternion,
                     cudaStream_t stream = 0);
