// adaptive_density.hpp

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief Launch CUDA kernel to perform adaptive density step
 *
 * @param[in] N Total number of gaussians
 * @param[in] iter Iteration number in trainer
 * @param[in] num_sh_coef Total number of spherical harmonics coefficients
 * @param[in] stream The CUDA stream to execute on
 * @param[in] use_adaptive_fractional_densification Flag to enable adaptive fractional density control
 * @param[in] adaptive_control_end Last interation number to run adaptive density
 * @param[in] adaptive_control_start First iteration to run adaptive density
 * @param[in] uv_grad_threshold UV gradeint threshold
 * @param[in] use_fractional_densification Flag to enable fractional density control
 * @param[in] uv_grad_percentile Quantile percent to split gaussians by
 * @param[[TODO:direction]] scale_norm_percentile [TODO:description]
 * @param[[TODO:direction]] max_gaussians [TODO:description]
 * @param[[TODO:direction]] use_delete [TODO:description]
 * @param[[TODO:direction]] use_clone [TODO:description]
 * @param[[TODO:direction]] use_split [TODO:description]
 * @param[[TODO:direction]] delete_opacity_threshold [TODO:description]
 * @param[[TODO:direction]] clone_scale_threshold [TODO:description]
 * @param[[TODO:direction]] num_split_samples [TODO:description]
 * @param[[TODO:direction]] split_scale_factor [TODO:description]
 * @param[[TODO:direction]] uv_grad_accum [TODO:description]
 * @param[[TODO:direction]] grad_accum_count [TODO:description]
 * @param[[TODO:direction]] scale [TODO:description]
 * @param[[TODO:direction]] d_mask [TODO:description]
 * @param[[TODO:direction]] xyz_grad_accum [TODO:description]
 * @param[[TODO:direction]] xyz [TODO:description]
 * @param[[TODO:direction]] rgb [TODO:description]
 * @param[[TODO:direction]] sh [TODO:description]
 * @param[[TODO:direction]] opacity [TODO:description]
 * @param[[TODO:direction]] quaternion [TODO:description]
 * @return [TODO:description]
 */
int adaptive_density(const int N, const int iter, const int num_sh_coef, cudaStream_t stream,
                     const bool use_adaptive_fractional_densification, const int adaptive_control_end,
                     const int adaptive_control_start, const float uv_grad_threshold,
                     const bool use_fractional_densification, const float uv_grad_percentile,
                     const float scale_norm_percentile, const int max_gaussians, const bool use_delete,
                     const bool use_clone, const bool use_split, const float delete_opacity_threshold,
                     const float clone_scale_threshold, const int num_split_samples, const float split_scale_factor,
                     const float *uv_grad_accum, const int *grad_accum_count, float *scale, bool *d_mask,
                     const float *xyz_grad_accum, float *xyz, float *rgb, float *sh, float *opacity, float *quaternion);
