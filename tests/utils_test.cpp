#include "gsplat/utils.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>

// Test fixture for config parsing tests.
// This fixture handles the creation and cleanup of temporary YAML files needed for testing.
class ConfigTest : public ::testing::Test {
protected:
  // This method is called before each test in this suite.
  void SetUp() override {
    // Create a valid YAML config file for testing successful parsing.
    valid_config_path = std::filesystem::temp_directory_path() / "valid_config.yaml";
    std::ofstream out(valid_config_path);
    out << "dataset_path: \"/data/nerf_synthetic/lego\"\n"
        << "output_dir: \"/output/lego\"\n"
        << "downsample_factor: 2\n"
        << "print_interval: 100\n"
        << "num_iters: 30000\n"
        << "ssim_frac: 0.2\n"
        << "test_eval_interval: 1000\n"
        << "test_split_ratio: 8\n"
        << "initial_opacity: 0.1\n"
        << "initial_scale_num_neighbors: 3\n"
        << "initial_scale_factor: 1.0\n"
        << "max_initial_scale: 1.0\n"
        << "near_thresh: 0.01\n"
        << "mh_dist: 1000.0\n"
        << "cull_mask_padding: 1\n"
        << "base_lr: 0.001\n"
        << "xyz_lr_multiplier_init: 1.0\n"
        << "xyz_lr_multiplier_final: 1.0\n"
        << "quat_lr_multiplier: 1.0\n"
        << "scale_lr_multiplier: 1.0\n"
        << "opacity_lr_multiplier: 1.0\n"
        << "rgb_lr_multiplier: 1.0\n"
        << "sh_lr_multiplier: 1.0\n"
        << "use_background: true\n"
        << "use_background_end: 15000\n"
        << "reset_opacity_interval: 3000\n"
        << "reset_opacity_value: 0.01\n"
        << "reset_opacity_start: 4000\n"
        << "reset_opacity_end: 15000\n"
        << "use_sh_precompute: true\n"
        << "max_sh_band: 2\n"
        << "add_sh_band_interval: 1000\n"
        << "use_split: true\n"
        << "use_clone: true\n"
        << "use_delete: true\n"
        << "adaptive_control_start: 500\n"
        << "adaptive_control_end: 20000\n"
        << "adaptive_control_interval: 100\n"
        << "max_gaussians: 1000000\n"
        << "delete_opacity_threshold: 0.005\n"
        << "uv_grad_threshold: 0.0002\n"
        << "split_scale_factor: 1.5\n";
    out.close();

    // Create a YAML file that is missing a required key.
    missing_key_config_path = std::filesystem::temp_directory_path() / "missing_key.yaml";
    std::ofstream missing_key_out(missing_key_config_path);
    missing_key_out << "output_dir: \"/output/lego\"\n";
    // "dataset_path" is intentionally omitted.
    missing_key_out.close();
  }

  // This method is called after each test in this suite.
  void TearDown() override {
    // Clean up the temporary files.
    std::filesystem::remove(valid_config_path);
    std::filesystem::remove(missing_key_config_path);
  }

  std::filesystem::path valid_config_path;
  std::filesystem::path missing_key_config_path;
};

// Test that a valid config file is parsed correctly.
TEST_F(ConfigTest, ParseValidConfig) {
  const ConfigParameters params = parseConfig(valid_config_path.string());

  // Verify a few key parameters to ensure the file was read correctly.
  EXPECT_EQ(params.dataset_path, "/data/nerf_synthetic/lego");
  EXPECT_EQ(params.downsample_factor, 2);
  EXPECT_FLOAT_EQ(params.ssim_frac, 0.2);
  EXPECT_TRUE(params.use_background);
  EXPECT_EQ(params.max_sh_band, 2);
  EXPECT_FLOAT_EQ(params.split_scale_factor, 1.5);
}

// Test that parsing a non-existent file throws an exception.
TEST_F(ConfigTest, ThrowsOnFileNotExist) {
  EXPECT_THROW(
      {
        try {
          parseConfig("non_existent_file.yaml");
        } catch (const std::runtime_error &e) {
          throw;
        }
      },
      std::runtime_error);
}

// Test that parsing a file with a missing key throws an exception.
TEST_F(ConfigTest, ThrowsOnMissingKey) {
  EXPECT_THROW(
      {
        try {
          parseConfig(missing_key_config_path.string());
        } catch (const std::runtime_error &e) {
          throw;
        }
      },
      std::runtime_error);
}

TEST(PlyUtilsTest, SavePly) {
  // Create dummy Gaussians
  Gaussians g;
  g.xyz.push_back(Eigen::Vector3f(1.0f, 2.0f, 3.0f));
  g.rgb.push_back(Eigen::Vector3f(0.5f, 0.5f, 0.5f));   // Should result in f_dc = 0
  g.opacity.push_back(0.0f);                            // Logit
  g.scale.push_back(Eigen::Vector3f(0.0f, 0.0f, 0.0f)); // Log scale
  g.quaternion.push_back(Eigen::Quaternionf(1.0f, 0.0f, 0.0f, 0.0f));

  // Add SH
  std::vector<Eigen::VectorXf> sh_vec;
  Eigen::VectorXf sh_coeffs(3);
  sh_coeffs << 0.1f, 0.2f, 0.3f;
  sh_vec.push_back(sh_coeffs);
  g.sh = sh_vec;

  std::string filename = "test_output.ply";
  save_ply(g, filename);

  // Verify file exists
  ASSERT_TRUE(std::filesystem::exists(filename));

  // Verify file size > 0
  ASSERT_GT(std::filesystem::file_size(filename), 0);

  // Basic header check
  std::ifstream infile(filename, std::ios::binary);
  std::string line;
  std::getline(infile, line);
  EXPECT_EQ(line, "ply");

  // Cleanup
  std::filesystem::remove(filename);
}
