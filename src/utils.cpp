#include "gsplat/utils.hpp"
#include "yaml-cpp/yaml.h"
#include <stdexcept>

// Helper function to safely get a value from a YAML node.
// Throws a runtime_error if the key is not found.
template <typename T> T getNodeValue(const YAML::Node &node, const std::string &key) {
  if (!node[key]) {
    throw std::runtime_error("Missing required parameter in YAML file: " + key);
  }
  return node[key].as<T>();
}

ConfigParameters parseConfig(const std::string &filename) {
  ConfigParameters params;
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // File paths and directories
    params.dataset_path = getNodeValue<std::string>(config, "dataset_path");
    params.output_dir = getNodeValue<std::string>(config, "output_dir");

    // General settings
    params.downsample_factor = getNodeValue<int>(config, "downsample_factor");
    params.print_interval = getNodeValue<int>(config, "print_interval");
    params.num_iters = getNodeValue<int>(config, "num_iters");
    params.ssim_frac = getNodeValue<float>(config, "ssim_frac");
    params.test_eval_interval = getNodeValue<int>(config, "test_eval_interval");
    params.test_split_ratio = getNodeValue<int>(config, "test_split_ratio");

    // Initial Gaussian properties
    params.initial_opacity = getNodeValue<float>(config, "initial_opacity");
    params.initial_scale_num_neighbors = getNodeValue<int>(config, "initial_scale_num_neighbors");
    params.initial_scale_factor = getNodeValue<float>(config, "initial_scale_factor");
    params.max_initial_scale = getNodeValue<float>(config, "max_initial_scale");

    // Rendering thresholds
    params.near_thresh = getNodeValue<float>(config, "near_thresh");
    params.far_thresh = getNodeValue<float>(config, "far_thresh");
    params.mh_dist = getNodeValue<float>(config, "mh_dist");
    params.cull_mask_padding = getNodeValue<int>(config, "cull_mask_padding");

    // Learning rates
    params.base_lr = getNodeValue<float>(config, "base_lr");
    params.xyz_lr_multiplier = getNodeValue<float>(config, "xyz_lr_multiplier");
    params.quat_lr_multiplier = getNodeValue<float>(config, "quat_lr_multiplier");
    params.scale_lr_multiplier = getNodeValue<float>(config, "scale_lr_multiplier");
    params.opacity_lr_multiplier = getNodeValue<float>(config, "opacity_lr_multiplier");
    params.rgb_lr_multiplier = getNodeValue<float>(config, "rgb_lr_multiplier");
    params.sh_lr_multiplier = getNodeValue<float>(config, "sh_lr_multiplier");

    // Background settings
    params.use_background = getNodeValue<bool>(config, "use_background");
    params.use_background_end = getNodeValue<int>(config, "use_background_end");

    // Opacity reset settings
    params.reset_opacity_interval = getNodeValue<int>(config, "reset_opacity_interval");
    params.reset_opacity_value = getNodeValue<float>(config, "reset_opacity_value");
    params.reset_opacity_start = getNodeValue<int>(config, "reset_opacity_start");
    params.reset_opacity_end = getNodeValue<int>(config, "reset_opacity_end");

    // Spherical Harmonics (SH) settings
    params.use_sh_precompute = getNodeValue<bool>(config, "use_sh_precompute");
    params.max_sh_band = getNodeValue<int>(config, "max_sh_band");
    params.add_sh_band_interval = getNodeValue<int>(config, "add_sh_band_interval");

    // Densification control
    params.use_split = getNodeValue<bool>(config, "use_split");
    params.use_clone = getNodeValue<bool>(config, "use_clone");
    params.use_delete = getNodeValue<bool>(config, "use_delete");
    params.adaptive_control_start = getNodeValue<int>(config, "adaptive_control_start");
    params.adaptive_control_end = getNodeValue<int>(config, "adaptive_control_end");
    params.adaptive_control_interval = getNodeValue<int>(config, "adaptive_control_interval");
    params.max_gaussians = getNodeValue<int>(config, "max_gaussians");
    params.delete_opacity_threshold = getNodeValue<float>(config, "delete_opacity_threshold");
    params.clone_scale_threshold = getNodeValue<float>(config, "clone_scale_threshold");
    params.max_scale_norm = getNodeValue<float>(config, "max_scale_norm");
    params.use_fractional_densification = getNodeValue<bool>(config, "use_fractional_densification");
    params.use_adaptive_fractional_densification = getNodeValue<bool>(config, "use_adaptive_fractional_densification");
    params.uv_grad_percentile = getNodeValue<float>(config, "uv_grad_percentile");
    params.scale_norm_percentile = getNodeValue<float>(config, "scale_norm_percentile");
    params.uv_grad_threshold = getNodeValue<float>(config, "uv_grad_threshold");
    params.split_scale_factor = getNodeValue<float>(config, "split_scale_factor");
    params.num_split_samples = getNodeValue<int>(config, "num_split_samples");

  } catch (const YAML::Exception &e) {
    // Re-throw as a standard exception for the caller to handle.
    throw std::runtime_error("Failed to parse YAML file '" + filename + "': " + e.what());
  }
  return params;
}
