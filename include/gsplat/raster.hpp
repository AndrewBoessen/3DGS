// raster.cpp

#pragma once

#include "dataloader/colmap.hpp"
#include "gsplat/gaussian.hpp"
#include "gsplat/utils.hpp"

/**
 * @brief Rasterize an image from a set of Gaussians.
 * @param[in] config      Configuration parameters for rendering.
 * @param[in] gaussians   The set of Gaussian splats to be rendered.
 * @param[in] image       The camera view information for the current image.
 * @param[in] camera      The camera model and intrinsic parameters.
 * @param[out] out_image  A host pointer to store the final rendered image.
 */
void rasterize_image(ConfigParameters config, Gaussians gaussians, Image image, Camera camera, float *out_image);
