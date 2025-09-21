// loss.hpp

#pragma once

#include <Eigen/Dense>
#include <math.h>

/**
 * @brief Computes the L1 loss for a multi-channel (e.g., RGB) image.
 *
 * @param predicted_data Pointer to the interleaved (e.g., RGBRGB...) predicted data.
 * @param gt_data Pointer to the interleaved ground truth data.
 * @param rows The height of the image.
 * @param cols The width of the image.
 * @param channels The number of channels (e.g., 3 for RGB).
 * @return The L1 loss value, averaged across all channels.
 */
float l1_loss(const float *predicted_data, const float *gt_data, int rows, int cols, int channels);

/**
 * @brief Computes the SSIM loss for a multi-channel (e.g., RGB) image.
 *
 * @param predicted_data Pointer to the interleaved (e.g., RGBRGB...) predicted data.
 * @param gt_data Pointer to the interleaved ground truth data.
 * @param rows The height of the image.
 * @param cols The width of the image.
 * @param channels The number of channels (e.g., 3 for RGB).
 * @return The SSIM loss value, averaged across all channels.
 */
float ssim_loss(const float *predicted_data, const float *gt_data, int rows, int cols, int channels);
