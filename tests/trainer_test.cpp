#include "gtest/gtest.h"

#define private public

#include "dataloader/colmap.hpp"
#include "gsplat/gaussian.hpp"
#include "gsplat/trainer.hpp"
#include "gsplat/utils.hpp"

// Helper function to create a Gaussians object with sample data for testing.
Gaussians create_sample_gaussians_for_trainer(size_t count, bool with_sh = false, int sh_band = 0) {
  std::vector<Eigen::Vector3f> xyz, rgb, scale;
  std::vector<float> opacity;
  std::vector<Eigen::Quaternionf> quaternion;
  std::optional<std::vector<Eigen::VectorXf>> sh_opt = std::nullopt;

  xyz.reserve(count);
  rgb.reserve(count);
  scale.reserve(count);
  opacity.reserve(count);
  quaternion.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    float val = static_cast<float>(i + 1); // Use 1-based to avoid issues with 0
    xyz.emplace_back(val, val, val);
    rgb.emplace_back(val / 255.0f, val / 255.0f, val / 255.0f);
    opacity.push_back(0.1f * val);
    // Use log scale for scales, as this is a common representation
    scale.emplace_back(std::log(0.01f * val), std::log(0.02f * val), std::log(0.03f * val));
    quaternion.emplace_back(Eigen::Quaternionf(1.0f, 0.1f * val, 0.2f * val, 0.3f * val).normalized());
  }

  if (with_sh) {
    std::vector<Eigen::VectorXf> sh_vec;
    sh_vec.reserve(count);
    size_t sh_dim = (sh_band + 1) * (sh_band + 1);
    for (size_t i = 0; i < count; ++i) {
      sh_vec.push_back(Eigen::VectorXf::Constant(sh_dim, static_cast<float>(i) * 0.1f));
    }
    sh_opt = std::move(sh_vec);
  }

  return Gaussians(std::move(xyz), std::move(rgb), std::move(opacity), std::move(scale), std::move(quaternion),
                   std::move(sh_opt));
}

// Test Fixture for the Trainer class.
class TrainerTest : public ::testing::Test {
protected:
  ConfigParameters config;
  std::unordered_map<int, Image> images;
  std::unordered_map<int, Camera> cameras;

  void SetUp() override {
    config = ConfigParameters{};
    config.test_split_ratio = 8;
    config.reset_opacity_value = 0.2f;
    config.max_sh_band = 2;
    config.initial_opacity = 0.1f;
    config.split_scale_factor = 1.6f;
    config.num_split_samples = 2;

    // Setup 16 images for splitting tests. Names are padded for correct sorting.
    for (int i = 1; i <= 16; ++i) {
      Image img;
      img.id = i;
      img.name = std::string("img_") + (i < 10 ? "0" : "") + std::to_string(i) + ".png";
      img.camera_id = i;
      images[i] = img;
      cameras[i] = Camera{};
    }
  }
};

TEST_F(TrainerTest, TestTrainSplit) {
  // Case 1: Split ratio of 8. With 16 images, this means every 8th image
  // goes to the test set (images at index 0 and 8).
  config.test_split_ratio = 8;
  Trainer trainer(config, Gaussians{}, std::move(images), std::move(cameras));
  trainer.test_train_split();

  ASSERT_EQ(trainer.test_images.size(), 2);
  ASSERT_EQ(trainer.train_images.size(), 14);
  // Verify that the correct images were chosen based on sorted names
  ASSERT_EQ(trainer.test_images[0].name, "img_01.png");
  ASSERT_EQ(trainer.test_images[1].name, "img_09.png");
}

TEST_F(TrainerTest, TestTrainSplitEdgeCases) {
  // Case 1: Split ratio <= 0. All images should go to the training set.
  config.test_split_ratio = 0;
  Trainer trainer_all_train(config, Gaussians{}, std::move(images), std::move(cameras));
  trainer_all_train.test_train_split();
  ASSERT_EQ(trainer_all_train.test_images.size(), 0);
  ASSERT_EQ(trainer_all_train.train_images.size(), 16);

  // Case 2: Split ratio = 1. All images should go to the test set.
  SetUp();
  config.test_split_ratio = 1;
  Trainer trainer_all_test(config, Gaussians{}, std::move(images), std::move(cameras));
  trainer_all_test.test_train_split();
  ASSERT_EQ(trainer_all_test.test_images.size(), 16);
  ASSERT_EQ(trainer_all_test.train_images.size(), 0);
}

TEST_F(TrainerTest, TestTrainSplitEmpty) {
  // Test with no images. Should not crash and result in empty vectors.
  Trainer trainer(config, Gaussians{}, {}, {});
  trainer.test_train_split();
  ASSERT_TRUE(trainer.test_images.empty());
  ASSERT_TRUE(trainer.train_images.empty());
}

TEST_F(TrainerTest, ResetOpacity) {
  Gaussians gaussians = create_sample_gaussians_for_trainer(10);
  config.reset_opacity_value = 0.5f;
  Trainer trainer(config, std::move(gaussians), {}, {});

  trainer.reset_opacity();

  for (const auto &op : trainer.gaussians.opacity) {
    // New opacity should be inverse sigmoid of 0.5
    ASSERT_FLOAT_EQ(op, 0.0f);
  }
}

// NOTE: The add_sh_band function is currently empty in trainer.cpp. This test
// is written against a plausible, correct implementation. It will fail until
// the function is implemented.
TEST_F(TrainerTest, AddShBand) {
  Gaussians gaussians_no_sh = create_sample_gaussians_for_trainer(5, false);
  config.max_sh_band = 1;
  Trainer trainer(config, std::move(gaussians_no_sh), {}, {});

  // From no SH to SH band 0
  trainer.add_sh_band();
  ASSERT_TRUE(trainer.gaussians.sh.has_value());
  ASSERT_EQ(trainer.gaussians.sh->size(), 5);
  // Band 0 has (0+1)^2 = 1 coefficient
  ASSERT_EQ((*trainer.gaussians.sh)[0].size(), 1);

  // From SH band 0 to SH band 1
  trainer.add_sh_band();
  ASSERT_TRUE(trainer.gaussians.sh.has_value());
  // Band 1 has (1+1)^2 = 4 coefficients
  ASSERT_EQ((*trainer.gaussians.sh)[0].size(), 4);

  // Should not add bands beyond max_sh_band
  trainer.add_sh_band();
  ASSERT_EQ((*trainer.gaussians.sh)[0].size(), 4);
}

// NOTE: The clone_gaussians function is not implemented yet in trainer.cpp.
// It will fail until this function is implemented.
TEST_F(TrainerTest, CloneGaussians) {
  Gaussians gaussians = create_sample_gaussians_for_trainer(10);
  config.initial_opacity = 0.123f; // A distinct value for checking
  Trainer trainer(config, std::move(gaussians), {}, {});

  // Clone gaussians at indices 1, 4, 8
  std::vector<bool> clone_mask(10, false);
  clone_mask[1] = true;
  clone_mask[4] = true;
  clone_mask[8] = true;

  // Capture original values before cloning
  auto original_g1 = trainer.gaussians.xyz[1];
  auto original_g4 = trainer.gaussians.xyz[4];
  auto original_g8 = trainer.gaussians.xyz[8];

  trainer.clone_gaussians(clone_mask);

  // Size should increase by the number of clones
  ASSERT_EQ(trainer.gaussians.size(), 13);

  // Check that the new gaussians are copies of the originals
  ASSERT_TRUE(trainer.gaussians.xyz[10].isApprox(original_g1));
  ASSERT_TRUE(trainer.gaussians.xyz[11].isApprox(original_g4));
  ASSERT_TRUE(trainer.gaussians.xyz[12].isApprox(original_g8));

  // Check that the opacity of the new gaussians is reset
  ASSERT_FLOAT_EQ(trainer.gaussians.opacity[10], config.initial_opacity);
  ASSERT_FLOAT_EQ(trainer.gaussians.opacity[11], config.initial_opacity);
  ASSERT_FLOAT_EQ(trainer.gaussians.opacity[12], config.initial_opacity);
}

TEST_F(TrainerTest, SplitGaussians) {
  Gaussians gaussians = create_sample_gaussians_for_trainer(10);
  config.num_split_samples = 2;
  config.split_scale_factor = 1.6f;
  Trainer trainer(config, std::move(gaussians), {}, {});

  // Split gaussians at indices 2 and 5
  std::vector<bool> split_mask(10, false);
  split_mask[2] = true;
  split_mask[5] = true;

  auto original_scale_2 = trainer.gaussians.scale[2];
  size_t original_size = trainer.gaussians.size();
  size_t num_to_split = 2;

  trainer.split_gaussians(split_mask);

  // Originals are removed (2) and new samples are added (2 * 2 = 4)
  size_t expected_size = original_size - num_to_split + num_to_split * config.num_split_samples;
  ASSERT_EQ(trainer.gaussians.size(), expected_size);

  auto new_scale = trainer.gaussians.scale[8]; // First new gaussian from original index 2

  // The logic is: new_scale = log(exp(old_scale) / factor)
  Eigen::Vector3f expected_scale_vec = (original_scale_2.array().exp() / config.split_scale_factor).log();
  ASSERT_TRUE(new_scale.isApprox(expected_scale_vec));
}
