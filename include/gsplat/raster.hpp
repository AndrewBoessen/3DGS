// raster.cpp

#pragma once

#include "dataloader/colmap.hpp"
#include "gsplat/gaussian.hpp"

void rasterize_image(Gaussians gaussians, Image image, Camera camera);
