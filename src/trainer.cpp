// trainer.cpp

#include "gsplat/trainer.hpp"
#include "gsplat/gaussian.hpp"
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
  } else {
    // Use the "every N-th image" strategy for a representative split
    for (size_t i = 0; i < all_images.size(); ++i) {
      if (i % split == 0) {
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

void Trainer::split_gaussians(const std::vector<bool> &split_mask) {
  // the number of Guassians to split into
  const int num_samples = config.num_split_samples;

  std::vector<Eigen::Quaternionf> split_quat;
  std::vector<Eigen::Vector3f> split_scale;
  std::vector<float> split_opacity;
  std::vector<Eigen::Vector3f> split_rgb;
  std::vector<Eigen::Vector3f> split_xyz;
  std::vector<Eigen::VectorXf> split_sh;

  assert(split_mask.size() == gaussians.xyz.size());

  // Count how many Gaussians will be split to pre-allocate memory. This is an
  // optimization that prevents the vectors from having to resize multiple times.
  const size_t num_to_split = std::count(split_mask.begin(), split_mask.end(), true);
  const size_t total_samples = num_to_split * num_samples;

  split_quat.reserve(total_samples);
  split_scale.reserve(total_samples);
  split_opacity.reserve(total_samples);
  split_rgb.reserve(total_samples);
  split_xyz.reserve(total_samples);
  if (gaussians.sh.has_value()) {
    split_sh.reserve(total_samples);
  }

  // Apply mask to Gaussians
  for (size_t i = 0; i < split_mask.size(); ++i) {
    if (split_mask[i]) {
      // duplicate j times
      for (size_t j = 0; j < num_samples; ++j) {
        split_xyz.push_back(gaussians.xyz[i]);
        split_quat.push_back(gaussians.quaternion[i]);
        split_scale.push_back(gaussians.scale[i]);
        split_opacity.push_back(gaussians.opacity[i]);
        split_rgb.push_back(gaussians.rgb[i]);

        // Copy spherical harmonics coefficients if they are being used.
        if (gaussians.sh.has_value()) {
          split_sh.push_back(gaussians.sh.value()[i]);
        }
      }
    }
  }

  // Sample from Gaussians
  Eigen::Array<float, 3, Eigen::Dynamic> random_samples =
      (Eigen::Array<float, 3, Eigen::Dynamic>::Random(3, total_samples) + 1.0f) * 0.5f;
  // hack to remove 0.0 from samples
  auto is_zero_mask = (random_samples == 0.0f);
  random_samples = is_zero_mask.select(1.0f, random_samples);

  // Map std::vectors to Eigen objects for vectorized operations (no copy)
  Eigen::Map<Eigen::Array<float, 3, Eigen::Dynamic>> split_xyz_map((float *)split_xyz.data(), 3, total_samples);
  Eigen::Map<Eigen::Array<float, 3, Eigen::Dynamic>> split_scale_map((float *)split_scale.data(), 3, total_samples);

  // Scale the random samples by the exponential of the Gaussian scales
  // random_samples = random_samples * exp(split_scale)
  random_samples *= split_scale_map.exp();

  // Rotate the scaled samples
  for (size_t i = 0; i < total_samples; ++i) {
    // Normalize the quaternion
    split_quat[i].normalize();
    // Apply rotation to the i-th sample
    random_samples.col(i) = (split_quat[i].toRotationMatrix() * random_samples.col(i).matrix()).array();
  }

  // Translate the original means by the transformed random samples
  // split_xyz += random_samples
  split_xyz_map += random_samples;

  // Update the scales
  // split_scale = log(exp(split_scale) / config.split_scale_factor)
  split_scale_map = (split_scale_map.exp() / config.split_scale_factor).log();

  // Remove split Gaussians
  std::vector<bool> keep_mask;
  keep_mask.reserve(split_mask.size());
  std::transform(split_mask.begin(), split_mask.end(), std::back_inserter(keep_mask), std::logical_not<bool>());

  gaussians.filter(keep_mask);

  Gaussians new_gaussians(std::move(split_xyz), std::move(split_rgb), std::move(split_opacity), std::move(split_scale),
                          std::move(split_quat),
                          gaussians.sh.has_value() ? std::make_optional(std::move(split_sh)) : std::nullopt);
  gaussians.append(new_gaussians);
}

void Trainer::clone_gaussians(const std::vector<bool> &clone_mask) {}

void Trainer::adaptive_density() {}
