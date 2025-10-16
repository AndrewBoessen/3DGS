#pragma once

#include "dataloader/colmap.hpp"
#include "gsplat/cuda_data.hpp"
#include "gsplat/utils.hpp"

/**
 * @brief Rasterizes an image from a set of Gaussians using pre-allocated CUDA buffers.
 * This function is optimized for training loops as it avoids repeated memory allocation.
 *
 * @param[in]     num_gaussians  The total number of Gaussians.
 * @param[in]     camera         The camera model and intrinsic parameters.
 * @param[in]     config         Configuration parameters for rendering.
 * @param[in,out] cuda           A manager for long-lived CUDA device buffers.
 * @param[out]    pass_data      A struct to be populated with pointers to per-iteration device buffers
 */
void rasterize_image(const int num_gaussians, const Camera &camera, const ConfigParameters &config,
                     CudaDataManager &cuda, ForwardPassData &pass_data, float bg_color);
