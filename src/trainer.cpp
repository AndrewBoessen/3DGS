// trainer.cpp

#include "gsplat/trainer.hpp"
#include "gsplat/gaussian.hpp"
#include <cmath>

// Helper function to filter a vector based on a boolean mask.
template <typename T> void filter_vector(std::vector<T> &vec, const std::vector<bool> &keep_mask) {
  if (vec.empty())
    return;
  assert(vec.size() == keep_mask.size());
  std::vector<T> filtered_vec;
  filtered_vec.reserve(std::count(keep_mask.begin(), keep_mask.end(), true));
  for (size_t i = 0; i < vec.size(); ++i) {
    if (keep_mask[i]) {
      filtered_vec.push_back(std::move(vec[i]));
    }
  }
  vec = std::move(filtered_vec);
}

void Trainer::reset_grad_accum() {
  const int size = gaussians.size();

  uv_grad_accum.assign(size, Eigen::Vector2f::Zero());
  xyz_grad_accum.assign(size, Eigen::Vector3f::Zero());
  grad_accum_dur.assign(size, 0);
}

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

  // Reset gradient accumulators
  reset_grad_accum();
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

  assert(split_mask.size() == gaussians.size());

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

  // Update optimizer state
  optimizer.filter_states(keep_mask);
  optimizer.append_states(total_samples);
}

void Trainer::clone_gaussians(const std::vector<bool> &clone_mask, const std::vector<Eigen::Vector3f> &xyz_grad_avg) {

  std::vector<Eigen::Quaternionf> clone_quat;
  std::vector<Eigen::Vector3f> clone_scale;
  std::vector<float> clone_opacity;
  std::vector<Eigen::Vector3f> clone_rgb;
  std::vector<Eigen::Vector3f> clone_xyz;
  std::vector<Eigen::VectorXf> clone_sh;

  std::vector<Eigen::Vector3f> clone_xyz_grad_avg;

  assert(clone_mask.size() == gaussians.size());

  const size_t num_to_clone = std::count(clone_mask.begin(), clone_mask.end(), true);

  clone_quat.reserve(num_to_clone);
  clone_scale.reserve(num_to_clone);
  clone_opacity.reserve(num_to_clone);
  clone_rgb.reserve(num_to_clone);
  clone_xyz.reserve(num_to_clone);
  if (gaussians.sh.has_value()) {
    clone_sh.reserve(num_to_clone);
  }
  clone_xyz_grad_avg.reserve(num_to_clone);

  // Apply mask to Gaussians
  for (size_t i = 0; i < clone_mask.size(); ++i) {
    if (clone_mask[i]) {
      clone_quat.push_back(gaussians.quaternion[i]);
      clone_scale.push_back(gaussians.scale[i]);
      clone_opacity.push_back(gaussians.opacity[i]);
      clone_rgb.push_back(gaussians.rgb[i]);
      clone_xyz.push_back(gaussians.xyz[i]);
      if (gaussians.sh.has_value()) {
        clone_sh.push_back(gaussians.sh.value()[i]);
      }
      clone_xyz_grad_avg.push_back(xyz_grad_avg[i]);
    }
  }

  // Move cloned Gaussian means based on average gradients
  for (size_t i = 0; i < clone_xyz.size(); ++i) {
    clone_xyz[i] = clone_xyz[i] - (clone_xyz_grad_avg[i] * 0.01f);
  }

  // Initialize new cloned Gaussians
  Gaussians new_gaussians(std::move(clone_xyz), std::move(clone_rgb), std::move(clone_opacity), std::move(clone_scale),
                          std::move(clone_quat),
                          gaussians.sh.has_value() ? std::make_optional(std::move(clone_sh)) : std::nullopt);

  // Update gaussians and optimizer
  gaussians.append(new_gaussians);
  optimizer.append_states(num_to_clone);
}

void Trainer::adaptive_density() {
  if (!config.use_delete && !config.use_clone && !config.use_split) {
    return;
  }

  // --- Step 1: Delete Gaussians ---
  const float op_threshold = log(config.delete_opacity_threshold) - log(1.0f - config.delete_opacity_threshold);
  std::vector<bool> keep_mask(gaussians.size());
  size_t delete_count = 0;

  for (size_t i = 0; i < gaussians.size(); ++i) {
    const bool low_opacity = gaussians.opacity[i] < op_threshold;
    const bool not_viewed = grad_accum_dur[i] == 0;
    const bool zero_grad = uv_grad_accum[i].norm() == 0.0f;
    keep_mask[i] = !(low_opacity || not_viewed || zero_grad);
    if (!keep_mask[i]) {
      delete_count++;
    }
  }

  if (delete_count > 0 && config.use_delete) {
    gaussians.filter(keep_mask);
    optimizer.filter_states(keep_mask);
    // Filter gradient accumulators to keep them in sync
    filter_vector(uv_grad_accum, keep_mask);
    filter_vector(xyz_grad_accum, keep_mask);
    filter_vector(grad_accum_dur, keep_mask);
  }

  // --- Check if max Gaussians are exceeded before densification ---
  if (gaussians.size() > config.max_gaussians) {
    reset_grad_accum();
    return;
  }

  // --- Step 2: Densify Gaussians (Clone and Split) ---

  // Calculate average gradients
  std::vector<Eigen::Vector3f> xyz_grad_avg(gaussians.size());
  std::vector<Eigen::Vector2f> uv_grad_avg(gaussians.size());
  for (size_t i = 0; i < gaussians.size(); ++i) {
    if (grad_accum_dur[i] > 0) {
      xyz_grad_avg[i] = xyz_grad_accum[i] / grad_accum_dur[i];
      uv_grad_avg[i] = uv_grad_accum[i] / grad_accum_dur[i];
    } else {
      xyz_grad_avg[i].setZero();
      uv_grad_avg[i].setZero();
    }
  }

  // Calculate norm of UV gradients
  std::vector<float> uv_grad_avg_norm(gaussians.size());
  std::transform(uv_grad_avg.begin(), uv_grad_avg.end(), uv_grad_avg_norm.begin(),
                 [](const Eigen::Vector2f &v) { return v.norm(); });

  // Determine the densification threshold
  float uv_split_val;
  if (config.use_fractional_densification) {
    float scale_factor = 1.0f;
    if (config.use_adaptive_fractional_densification) {
      scale_factor = std::max(0.0f, (static_cast<float>(config.adaptive_control_end - iter) /
                                     static_cast<float>(config.adaptive_control_end - config.adaptive_control_start)) *
                                        2.0f);
    }
    float uv_percentile = 1.0f - (1.0f - config.uv_grad_percentile) * scale_factor;
    std::vector<float> sorted_norms = uv_grad_avg_norm;
    std::sort(sorted_norms.begin(), sorted_norms.end());
    int index = std::min(static_cast<int>(uv_percentile * (sorted_norms.size() - 1)), (int)sorted_norms.size() - 1);
    uv_split_val = sorted_norms[index];
  } else {
    uv_split_val = config.uv_grad_threshold;
  }

  // Identify Gaussians to densify
  std::vector<bool> densify_mask(gaussians.size());
  size_t densify_count = 0;
  for (size_t i = 0; i < gaussians.size(); ++i) {
    if (uv_grad_avg_norm[i] > uv_split_val) {
      densify_mask[i] = true;
      densify_count++;
    } else {
      densify_mask[i] = false;
    }
  }

  // Get max scale component for each Gaussian
  std::vector<float> scale_max(gaussians.size());
  for (size_t i = 0; i < gaussians.size(); ++i) {
    scale_max[i] = gaussians.scale[i].array().exp().maxCoeff();
  }

  // --- Step 2.1: Clone small Gaussians ---
  std::vector<bool> clone_mask(gaussians.size());
  size_t clone_count = 0;
  for (size_t i = 0; i < gaussians.size(); ++i) {
    clone_mask[i] = densify_mask[i] && (scale_max[i] <= config.clone_scale_threshold);
    if (clone_mask[i]) {
      clone_count++;
    }
  }

  if (clone_count > 0 && config.use_clone) {
    const size_t original_size = gaussians.size();
    clone_gaussians(clone_mask, xyz_grad_avg);

    // Append values for new Gaussians to keep vectors in sync for the split step
    for (size_t i = 0; i < original_size; ++i) {
      if (clone_mask[i]) {
        densify_mask.push_back(densify_mask[i]);
        scale_max.push_back(scale_max[i]);
      }
    }
  }

  // --- Step 2.2: Split large Gaussians ---
  const size_t current_gaussians_count = gaussians.size();
  std::vector<bool> split_mask(current_gaussians_count, false);

  // Split based on densification candidates that were too large to clone
  for (size_t i = 0; i < current_gaussians_count; ++i) {
    if (densify_mask[i] && (scale_max[i] > config.clone_scale_threshold)) {
      split_mask[i] = true;
    }
  }

  // Split any Gaussians that are "too big", regardless of gradient
  float scale_factor = 1.0f;
  if (config.use_adaptive_fractional_densification) {
    scale_factor = std::max(0.0f, (static_cast<float>(config.adaptive_control_end - iter) /
                                   static_cast<float>(config.adaptive_control_end - config.adaptive_control_start)) *
                                      2.0f);
  }
  float scale_percentile = 1.0f - (1.0f - config.scale_norm_percentile) * scale_factor;
  std::vector<float> sorted_scales = scale_max;
  std::sort(sorted_scales.begin(), sorted_scales.end());
  int index = std::min(static_cast<int>(scale_percentile * (sorted_scales.size() - 1)), (int)sorted_scales.size() - 1);
  float scale_split_val = sorted_scales[index];

  for (size_t i = 0; i < current_gaussians_count; ++i) {
    if (scale_max[i] > scale_split_val) {
      split_mask[i] = true;
    }
  }

  size_t split_count = std::count(split_mask.begin(), split_mask.end(), true);
  if (split_count > 0 && config.use_split) {
    split_gaussians(split_mask);
  }

  reset_grad_accum();
}
