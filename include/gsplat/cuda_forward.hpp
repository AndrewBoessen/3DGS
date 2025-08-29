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
