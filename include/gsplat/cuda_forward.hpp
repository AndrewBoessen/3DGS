// cuda_functions.hpp

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief Launches the CUDA kernel for projecting 3D points to 2D image coordinates.
 * @param[in]  xyz  A device pointer to the input array of 3D points.
 * @param[in]  K    A device pointer to the camera intrinsic matrix.
 * @param[in]  N    The total number of points.
 * @param[out] uv   A device pointer to the output array for 2D coordinates.
 */
void camera_intrinsic_projection(float *const xyz, const float *K, const int N, float *uv);

/**
 * @brief Lauches CUDA kernel to perform frustum culling on guassians.
 * @param[in]  uv           A device pointer to gaussian coordinates in image frame
 * @param[in]  xyz          A device pointer to 3D corrdinates of gaussians in camera perspective
 * @param[in]  N            Number of Gaussians
 * @param[in]  near_thresh  Distance to cull guassians closer than threshold
 * @param[in]  far_thresh   Distance to cull gaussians farther than threshold
 * @param[in]  padding      Padding distance beyond image frame
 * @param[in]  width        Image width
 * @param[in]  height       Image height
 * @param[out] mask         A device pointer for mask on guassians
 */
void cull_gaussians(float *const uv, float *const xyz, const int N, const float near_thresh, const float far_thresh,
                    const int padding, const int width, const int height, bool *mask);
