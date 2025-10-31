// adaptive_density.hpp

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief Clones the Gaussians specified in the mask
 * @param[in]   N            Total number of of Gaussians
 * @param[in]   num_sh_coef  Total number of SH coefficients
 * @param[in]   mask         Clone mask on all N Gaussians
 * @param[in]   write_ids    Write ids into output (exclusive sum on clone mask)
 * @param[in]   xyz_grad     Gradient accumulator for xyz positions
 * @param[in]   accum_dur    Number of iterations in accum
 * @param[in]   xyz_in       XYZ for all current Gaussians
 * @param[in]   rgb_in       RGB for all current Gaussians
 * @param[in]   op_in        Opacity for all current Gaussians
 * @param[in]   scale_in     Scale vectors for all current Gaussians
 * @param[in]   quat_in      Quaternion for all current Gaussians
 * @param[in]   sh_in        Spherical harmonics coefficients
 * @param[out]  xyz_out      XYZ output buffer
 * @param[out]  rgb_out      RGB output buffer
 * @param[out]  op_out       Opacity output buffer
 * @param[out]  scale_out    Scale output buffer
 * @param[out]  quat_out     Quaternion output buffer
 * @param[out]  sh_out       Spherical harmonics output buffer
 * @param[in]   stream       CUDA stream to execute on
 */
void clone_gaussians(const int N, const int num_sh_coef, const bool *mask, const int *write_ids, const float *xyz_grad,
                     const int *accum_dur, const float *xyz_in, const float *rgb_in, const float *op_in,
                     const float *scale_in, const float *quat_in, const float *sh_in, float *xyz_out, float *rgb_out,
                     float *op_out, float *scale_out, float *quat_out, float *sh_out, cudaStream_t stream = 0);

/**
 * @brief Splits the of Gaussians specified in the mask
 * @param[in]   N            Total number of of Gaussians
 * @param[in]   scale_factor Factor to divide split Gaussians scale by
 * @param[in]   num_sh_coef  Total number of SH coefficients
 * @param[in]   mask         Clone mask on all N Gaussians
 * @param[in]   write_ids    Write ids into output (exclusive sum on clone mask)
 * @param[in]   xyz_in       XYZ for all current Gaussians
 * @param[in]   rgb_in       RGB for all current Gaussians
 * @param[in]   op_in        Opacity for all current Gaussians
 * @param[in]   scale_in     Scale vectors for all current Gaussians
 * @param[in]   quat_in      Quaternion for all current Gaussians
 * @param[in]   sh_in        Spherical harmonics coefficients
 * @param[out]  xyz_out      XYZ output buffer
 * @param[out]  rgb_out      RGB output buffer
 * @param[out]  op_out       Opacity output buffer
 * @param[out]  scale_out    Scale output buffer
 * @param[out]  quat_out     Quaternion output buffer
 * @param[out]  sh_out       Spherical harmonics output buffer
 * @param[in]   stream       CUDA stream to execute on
 */
void split_gaussians(const int N, const float scale_factor, const int num_sh_coef, const bool *mask,
                     const int *write_ids, const float *xyz_in, const float *rgb_in, const float *op_in,
                     const float *scale_in, const float *quat_in, const float *sh_in, float *xyz_out, float *rgb_out,
                     float *op_out, float *scale_out, float *quat_out, float *sh_out, cudaStream_t stream = 0);
