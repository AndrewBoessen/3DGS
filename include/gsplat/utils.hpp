// utils.hpp

#pragma once

#include <string>

// This struct holds all the configuration parameters read from the YAML file.
// It provides a typed and organized way to access the settings throughout your application.
struct ConfigParameters {
  // File paths and directories
  std::string dataset_path;
  std::string output_dir;

  // General settings
  int downsample_factor;
  int print_interval;
  int num_iters;
  float ssim_frac;
  int test_eval_interval;
  int test_split_ratio;

  // Initial Gaussian properties
  float initial_opacity;
  int initial_scale_num_neighbors;
  float initial_scale_factor;
  float max_initial_scale;

  // Rendering thresholds
  float near_thresh;
  float far_thresh;
  float mh_dist;
  int cull_mask_padding;

  // Learning rates
  float base_lr;
  float xyz_lr_multiplier;
  float quat_lr_multiplier;
  float scale_lr_multiplier;
  float opacity_lr_multiplier;
  float rgb_lr_multiplier;
  float sh_lr_multiplier;

  // Background settings
  bool use_background;
  int use_background_end;

  // Opacity reset settings
  int reset_opacity_interval;
  float reset_opacity_value;
  int reset_opacity_start;
  int reset_opacity_end;

  // Spherical Harmonics (SH) settings
  bool use_sh_precompute;
  int max_sh_band;
  int add_sh_band_interval;

  // Densification control
  bool use_split;
  bool use_clone;
  bool use_delete;
  int adaptive_control_start;
  int adaptive_control_end;
  int adaptive_control_interval;
  int max_gaussians;
  float delete_opacity_threshold;
  float uv_grad_threshold;
  float percent_dense;
  float split_scale_factor;
  int num_split_samples;
};

/**
 * @brief Parses a YAML configuration file and loads parameters into a struct.
 *
 * This function reads the specified YAML file and populates a ConfigParameters
 * struct. It will throw an exception if the file cannot be found or if there's
 * a parsing error.
 *
 * @param filename The path to the YAML configuration file.
 * @return A ConfigParameters struct containing the parsed values.
 */
ConfigParameters parseConfig(const std::string &filename);
