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
  double ssim_frac;
  int test_eval_interval;
  int test_split_ratio;

  // Initial Gaussian properties
  double initial_opacity;
  int initial_scale_num_neighbors;
  double initial_scale_factor;
  double max_initial_scale;

  // Rendering thresholds
  double near_thresh;
  double far_thresh;
  double mh_dist;
  int cull_mask_padding;

  // Learning rates
  double base_lr;
  double xyz_lr_multiplier_init;
  double xyz_lr_multiplier_final;
  double quat_lr_multiplier;
  double scale_lr_multiplier;
  double opacity_lr_multiplier;
  double rgb_lr_multiplier;
  double sh_lr_multiplier;

  // Background settings
  bool use_background;
  int use_background_end;

  // Opacity reset settings
  int reset_opacity_interval;
  double reset_opacity_value;
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
  double delete_opacity_threshold;
  double uv_grad_threshold;
  double split_scale_factor;
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
