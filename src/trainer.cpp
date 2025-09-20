// trainer.cpp

#include "gsplat/trainer.hpp"
#include <cmath>

void Trainer::test_train_split() {
  const int split = config.test_split_ratio;

  // Ensure the destination vectors are empty before splitting
  test_images.clear();
  train_images.clear();

  if (images.empty()) {
    return;
  }

  // Convert the map of images to a vector to allow for sorting
  std::vector<Image> all_images;
  all_images.reserve(images.size());
  for (const auto &pair : images) {
    all_images.push_back(pair.second);
  }

  // Sort the images by name to ensure a deterministic split across runs
  std::sort(all_images.begin(), all_images.end(), [](const Image &a, const Image &b) { return a.name < b.name; });

  // Handle edge cases for the split ratio
  if (split <= 0) {
    train_images = all_images;
  } else if (split >= 1) {
    test_images = all_images;
  } else {
    // Use the "every N-th image" strategy for a representative split
    int step = static_cast<int>(std::round(1.0f / split));
    for (size_t i = 0; i < all_images.size(); ++i) {
      if (i % step == 0) {
        test_images.push_back(all_images[i]);
      } else {
        train_images.push_back(all_images[i]);
      }
    }
  }
}

void Trainer::reset_opacity() {
  // set all Gaussian opacity
  const double opc = config.reset_opacity_value;
  const double new_opc = log(opc) - log(1 - opc);
  std::fill(gaussians.opacity.begin(), gaussians.opacity.end(), new_opc);
}

void Trainer::add_sh_band() {
  size_t num_gaussians = gaussians.xyz.size();
  if (num_gaussians == 0) {
    return;
  }

  // Case 1: No SH coefficients exist, and the config allows for at least band 1.
  if (!gaussians.sh.has_value() && config.max_sh_band > 0) {
    // Initialize with coefficients for SH band 1.
    const size_t sh_coeffs_count = 3 * 3; // 3 coeffs * 3 channels

    gaussians.sh.emplace(num_gaussians, Eigen::VectorXf::Zero(sh_coeffs_count));
    optimizer.upgrade_sh_states(num_gaussians, sh_coeffs_count);
    return;
  }

  if (gaussians.sh.has_value()) {
    auto &sh_vec = gaussians.sh.value();
    if (sh_vec.empty())
      return;

    size_t old_total_coeffs = sh_vec[0].size();
    size_t old_coeffs_per_channel = old_total_coeffs / 3;

    // Case 2: Current band is 1 (3 coeffs/channel), and config allows for band 2.
    if (old_coeffs_per_channel == 3 && config.max_sh_band > 1) {
      // Upgrade from band 1 to band 2.
      // Band 2 requires (2+1)^2 - 1 = 8 coefficients per color channel.
      const size_t new_sh_coeffs_count = 8 * 3; // 8 coeffs * 3 channels

      for (auto &sh : sh_vec) {
        sh.conservativeResize(new_sh_coeffs_count);
        sh.tail(new_sh_coeffs_count - old_total_coeffs).setZero();
      }
      optimizer.upgrade_sh_states(num_gaussians, new_sh_coeffs_count);
    }
    // Case 3: Current band is 2 (8 coeffs/channel), and config allows for band 3.
    else if (old_coeffs_per_channel == 8 && config.max_sh_band > 2) {
      // Upgrade from band 2 to band 3.
      // Band 3 requires (3+1)^2 - 1 = 15 coefficients per color channel.
      const size_t new_sh_coeffs_count = 15 * 3; // 15 coeffs * 3 channels

      for (auto &sh : sh_vec) {
        sh.conservativeResize(new_sh_coeffs_count);
        sh.tail(new_sh_coeffs_count - old_total_coeffs).setZero();
      }
      optimizer.upgrade_sh_states(num_gaussians, new_sh_coeffs_count);
    }
  }
}
