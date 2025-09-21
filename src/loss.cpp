// loss.cpp

#include "gsplat/loss.hpp"

float l1_loss(const float *predicted_data, const float *gt_data, int rows, int cols, int channels) {
  // For L1 loss, we can treat the entire data block as a flat array, which is simpler
  // and more efficient than striding through each channel.
  long total_size = static_cast<long>(rows) * cols * channels;
  Eigen::Map<const Eigen::VectorXf> pred(predicted_data, total_size);
  Eigen::Map<const Eigen::VectorXf> gt(gt_data, total_size);
  return (pred - gt).array().abs().mean();
}

float ssim_loss(const float *predicted_data, const float *gt_data, int rows, int cols, int channels) {
  if (channels == 0)
    return 0.0f;

  float total_ssim_index = 0.0f;
  for (int c = 0; c < channels; ++c) {
    // Create a non-copying map for the current channel using strides
    Eigen::Map<const Eigen::MatrixXf, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> pred_c(
        predicted_data + c, rows, cols, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cols * channels, channels));
    Eigen::Map<const Eigen::MatrixXf, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> gt_c(
        gt_data + c, rows, cols, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(cols * channels, channels));

    // Calculate SSIM for the current channel
    const float K1 = 0.01f, K2 = 0.03f, L = 1.0f;
    const float C1 = (K1 * L) * (K1 * L), C2 = (K2 * L) * (K2 * L);
    float mu_p = pred_c.mean();
    float mu_gt = gt_c.mean();
    float sigma_p_sq = (pred_c.array() - mu_p).square().mean();
    float sigma_gt_sq = (gt_c.array() - mu_gt).square().mean();
    float sigma_p_gt = ((pred_c.array() - mu_p) * (gt_c.array() - mu_gt)).mean();
    float numerator = (2.0f * mu_p * mu_gt + C1) * (2.0f * sigma_p_gt + C2);
    float denominator = (mu_p * mu_p + mu_gt * mu_gt + C1) * (sigma_p_sq + sigma_gt_sq + C2);
    total_ssim_index += numerator / denominator;
  }

  float avg_ssim_index = total_ssim_index / static_cast<float>(channels);
  return 1.0f - avg_ssim_index;
}
