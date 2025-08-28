#include "dataloader/colmap.hpp"
#include "gsplat/gaussian.hpp"
#include "gtest/gtest.h"

// Helper function to create a Gaussians object with sample data for testing.
Gaussians create_sample_gaussians(size_t count, bool with_sh = false) {
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
    float val = static_cast<float>(i);
    xyz.emplace_back(val, val, val);
    rgb.emplace_back(val / 255.0f, val / 255.0f, val / 255.0f);
    opacity.push_back(0.1f * val);
    scale.emplace_back(0.01f * val, 0.01f * val, 0.01f * val);
    quaternion.emplace_back(Eigen::Quaternionf(1.0f, val, val, val).normalized());
  }

  if (with_sh) {
    std::vector<Eigen::VectorXf> sh_vec;
    sh_vec.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      Eigen::VectorXf sh_coeffs(3);
      sh_coeffs << static_cast<float>(i), static_cast<float>(i + 1), static_cast<float>(i + 2);
      sh_vec.push_back(sh_coeffs);
    }
    sh_opt = std::move(sh_vec);
  }
  return Gaussians(std::move(xyz), std::move(rgb), std::move(opacity), std::move(scale), std::move(quaternion),
                   std::move(sh_opt));
}

// ===================================================================
// Standalone Tests (for static methods or simple constructors)
// ===================================================================
class GaussiansStandaloneTest : public ::testing::Test {};

TEST_F(GaussiansStandaloneTest, DefaultConstructor) {
  const Gaussians g;
  EXPECT_EQ(g.size(), 0);
  EXPECT_TRUE(g.xyz.empty());
  EXPECT_FALSE(g.sh.has_value());
}

TEST_F(GaussiansStandaloneTest, Initialize) {
  std::unordered_map<uint64_t, Point3D> points;
  Point3D p1;
  p1.id = 1;
  p1.xyz = Eigen::Vector3d(1.0, 2.0, 3.0);
  p1.rgb = {128, 64, 32};
  points[p1.id] = p1;

  const Gaussians g = Gaussians::Initialize(points);
  EXPECT_EQ(g.size(), 1);
  EXPECT_FALSE(g.sh.has_value());
  EXPECT_TRUE(g.xyz[0].isApprox(Eigen::Vector3f(1.0f, 2.0f, 3.0f)));
  EXPECT_TRUE(g.rgb[0].isApprox(Eigen::Vector3f(128.0f / 255.0f, 64.0f / 255.0f, 32.0f / 255.0f)));
  EXPECT_FLOAT_EQ(g.opacity[0], 0.1f);
}

// ===================================================================
// Fixture-Based Tests (for methods on object instances)
// ===================================================================

// Test fixture that sets up common Gaussians objects for use in multiple tests.
class GaussiansTest : public ::testing::Test {
protected:
  // This method is called before each test in this suite.
  void SetUp() override {
    g_3_no_sh = create_sample_gaussians(3, false);
    g_5_with_sh = create_sample_gaussians(5, true);
  }

  Gaussians g_empty;
  Gaussians g_3_no_sh;
  Gaussians g_5_with_sh;
};

TEST_F(GaussiansTest, Append) {
  const auto g_3_copy = g_3_no_sh;
  Gaussians g_2_no_sh = create_sample_gaussians(2, false);

  g_2_no_sh.append(g_3_no_sh);
  EXPECT_EQ(g_2_no_sh.size(), 5);
  // Check that the last element is from the appended object
  EXPECT_TRUE(g_2_no_sh.xyz.back().isApprox(g_3_copy.xyz.back()));
}

TEST_F(GaussiansTest, AppendToEmpty) {
  const auto g_3_copy = g_3_no_sh;
  g_empty.append(g_3_no_sh);

  EXPECT_EQ(g_empty.size(), 3);
  EXPECT_TRUE(g_empty.xyz[0].isApprox(g_3_copy.xyz[0]));
}

TEST_F(GaussiansTest, AppendEmptyToNonEmpty) {
  const auto g_3_copy = g_3_no_sh;
  g_3_no_sh.append(g_empty);

  EXPECT_EQ(g_3_no_sh.size(), 3);
  EXPECT_TRUE(g_3_no_sh.xyz[0].isApprox(g_3_copy.xyz[0]));
}

TEST_F(GaussiansTest, AppendHandlesSHCorrectly) {
  // Case 1: Both have SH -> SH should be concatenated
  auto g_5_copy = g_5_with_sh;
  g_5_with_sh.append(g_5_copy);
  EXPECT_EQ(g_5_with_sh.size(), 10);
  ASSERT_TRUE(g_5_with_sh.sh.has_value());
  EXPECT_EQ(g_5_with_sh.sh->size(), 10);

  // Case 2: One has SH, one does not -> SH should be cleared
  g_5_with_sh.append(g_3_no_sh);
  EXPECT_EQ(g_5_with_sh.size(), 13);
  EXPECT_FALSE(g_5_with_sh.sh.has_value());
}

TEST_F(GaussiansTest, FilterKeepSome) {
  const auto g_5_copy = g_5_with_sh;
  const std::vector<bool> mask = {true, false, true, false, true};

  g_5_with_sh.filter(mask);

  EXPECT_EQ(g_5_with_sh.size(), 3);
  // Check that the correct elements were kept
  EXPECT_TRUE(g_5_with_sh.xyz[0].isApprox(g_5_copy.xyz[0]));
  EXPECT_TRUE(g_5_with_sh.xyz[1].isApprox(g_5_copy.xyz[2]));
  EXPECT_TRUE(g_5_with_sh.xyz[2].isApprox(g_5_copy.xyz[4]));

  // Check that SH was also filtered
  ASSERT_TRUE(g_5_with_sh.sh.has_value());
  EXPECT_TRUE((*g_5_with_sh.sh)[1].isApprox((*g_5_copy.sh)[2]));
}

TEST_F(GaussiansTest, FilterKeepNone) {
  const std::vector<bool> mask(g_3_no_sh.size(), false);
  g_3_no_sh.filter(mask);
  EXPECT_EQ(g_3_no_sh.size(), 0);
}

TEST_F(GaussiansTest, FilterKeepAll) {
  const auto g_3_copy = g_3_no_sh;
  const std::vector<bool> mask(g_3_no_sh.size(), true);
  g_3_no_sh.filter(mask);

  EXPECT_EQ(g_3_no_sh.size(), 3);
  EXPECT_TRUE(g_3_no_sh.xyz.back().isApprox(g_3_copy.xyz.back()));
}

TEST_F(GaussiansTest, FilterEmpty) {
  const std::vector<bool> mask;
  g_empty.filter(mask); // Should not crash
  EXPECT_EQ(g_empty.size(), 0);
}
