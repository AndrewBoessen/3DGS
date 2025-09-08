// raster.cpp

#pragma once

#include "dataloader/colmap.hpp"
#include "gsplat/gaussian.hpp"
#include "gsplat/utils.hpp"

void rasterize_image(ConfigParameters config, Gaussians gaussians, Image image, Camera camera, float *out_image);
