// loss_test.cpp

#include "gsplat/loss.hpp"
#include "gtest/gtest.h"

// Test Fixture for the Loss Functions.
class LossTest : public ::testing::Test {
protected:
  const int rows = 4;
  const int cols = 4;
  const int channels = 3;

  // Grayscale data
  std::vector<float> image_a_data;
  std::vector<float> image_b_data;

  // RGB data
  std::vector<float> rgb_a_data;
  std::vector<float> rgb_b_data;

  void SetUp() override {
    // Grayscale setup
    image_a_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
    image_b_data.resize(16);
    for (size_t i = 0; i < image_a_data.size(); ++i) {
      image_b_data[i] = image_a_data[i] + 0.1f;
    }

    // RGB setup
    rgb_a_data.resize(rows * cols * channels);
    rgb_b_data.resize(rows * cols * channels);
    for (int i = 0; i < rows * cols; ++i) {
      rgb_a_data[i * 3 + 0] = 0.2f; // R
      rgb_a_data[i * 3 + 1] = 0.4f; // G
      rgb_a_data[i * 3 + 2] = 0.6f; // B

      rgb_b_data[i * 3 + 0] = 0.3f; // R + 0.1
      rgb_b_data[i * 3 + 1] = 0.5f; // G + 0.1
      rgb_b_data[i * 3 + 2] = 0.7f; // B + 0.1
    }
  }
};

// --- L1 Loss Tests ---
TEST_F(LossTest, RgbL1LossIdentical) {
  float loss = l1_loss(rgb_a_data.data(), rgb_a_data.data(), rows, cols, channels);
  ASSERT_FLOAT_EQ(loss, 0.0f);
}

TEST_F(LossTest, RgbL1LossDifferent) {
  float loss = l1_loss(rgb_a_data.data(), rgb_b_data.data(), rows, cols, channels);
  ASSERT_NEAR(loss, 0.1f, 1e-6);
}

// --- SSIM Loss Tests ---
TEST_F(LossTest, RgbSsimLossIdentical) {
  float loss = ssim_loss(rgb_a_data.data(), rgb_a_data.data(), rows, cols, channels);
  ASSERT_NEAR(loss, 0.0f, 1e-6);
}

TEST_F(LossTest, RgbSsimLossDifferent) {
  float loss = ssim_loss(rgb_a_data.data(), rgb_b_data.data(), rows, cols, channels);
  ASSERT_GT(loss, 0.0f);
}
