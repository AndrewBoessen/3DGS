#include "gsplat/gaussian.hpp"
#include "gsplat/optimize.hpp"
#include "gtest/gtest.h"

// A helper function to create a Gaussians object for testing
Gaussians create_test_gaussians(size_t n, int sh_degree = -1) {
  std::vector<Eigen::Vector3f> xyz(n);
  std::vector<Eigen::Vector3f> rgb(n);
  std::vector<float> opacity(n, 0.5f);
  std::vector<Eigen::Vector3f> scale(n);
  std::vector<Eigen::Quaternionf> quaternion(n, Eigen::Quaternionf::Identity());
  std::optional<std::vector<Eigen::VectorXf>> sh = std::nullopt;

  for (size_t i = 0; i < n; ++i) {
    float val = static_cast<float>(i);
    xyz[i] = Eigen::Vector3f(val, val, val);
    rgb[i] = Eigen::Vector3f(0.1f * val, 0.1f * val, 0.1f * val);
    scale[i] = Eigen::Vector3f(1.0f + val, 1.0f + val, 1.0f + val);
  }

  if (sh_degree >= 0) {
    int sh_dim = (sh_degree + 1) * (sh_degree + 1);
    sh.emplace();
    sh->resize(n);
    for (size_t i = 0; i < n; ++i) {
      (*sh)[i] = Eigen::VectorXf::Ones(sh_dim) * 0.1f * static_cast<float>(i);
    }
  }

  return Gaussians(std::move(xyz), std::move(rgb), std::move(opacity), std::move(scale), std::move(quaternion),
                   std::move(sh));
}

// A helper function to create a Gradients object for testing
Gradients create_test_gradients(const Gaussians &params) {
  size_t n = params.size();
  Gradients grads;
  grads.xyz.resize(n);
  grads.rgb.resize(n);
  grads.opacity.resize(n);
  grads.scale.resize(n);
  grads.quaternion.resize(n);

  for (size_t i = 0; i < n; ++i) {
    float val = 0.1f * (i + 1);
    grads.xyz[i] = Eigen::Vector3f(val, -val, val);
    grads.rgb[i] = Eigen::Vector3f(-val, val, -val);
    grads.opacity[i] = val;
    grads.scale[i] = Eigen::Vector3f(val, val, -val);
    grads.quaternion[i] = Eigen::Vector3f(-val, val, val);
  }

  if (params.sh.has_value()) {
    grads.sh.emplace();
    grads.sh->resize(n);
    for (size_t i = 0; i < n; ++i) {
      (*grads.sh)[i] = Eigen::VectorXf::Ones((*params.sh)[i].size()) * 0.1f;
    }
  }
  return grads;
}

// Test fixture for AdamOptimizer tests
class AdamOptimizerTest : public ::testing::Test {
protected:
  AdamOptimizer optimizer;
  Gaussians params;
  Gradients grads;

  void SetUp() override {
    optimizer = AdamOptimizer(0.001f, 0.9f, 0.999f, 1e-8f);
    params = create_test_gaussians(3, 1); // 3 gaussians, SH degree 1
    grads = create_test_gradients(params);
  }
};

// Test that the optimizer is constructed with default values
TEST(AdamOptimizerConstruction, InitializesCorrectly) {
  AdamOptimizer opt;
  // We can't directly access private members, but we can test its behavior.
  // A step with zero gaussians should not crash.
  Gaussians empty_gaussians;
  Gradients empty_grads;
  ASSERT_NO_THROW(opt.step(empty_gaussians, empty_grads));
}

// Test that parameters do not change when gradients are zero
TEST_F(AdamOptimizerTest, StepWithZeroGradients) {
  Gaussians initial_params = create_test_gaussians(2);
  Gaussians params_to_update = create_test_gaussians(2);
  Gradients zero_grads = create_test_gradients(params_to_update);

  // Set all gradients to zero
  for (auto &g : zero_grads.xyz)
    g.setZero();
  for (auto &g : zero_grads.rgb)
    g.setZero();
  std::fill(zero_grads.opacity.begin(), zero_grads.opacity.end(), 0.0f);
  for (auto &g : zero_grads.scale)
    g.setZero();
  for (auto &g : zero_grads.quaternion)
    g.setZero();

  optimizer.step(params_to_update, zero_grads);

  for (size_t i = 0; i < initial_params.size(); ++i) {
    ASSERT_TRUE(initial_params.xyz[i].isApprox(params_to_update.xyz[i]));
  }
}

// Test the correctness of a single Adam update step for a single parameter
TEST_F(AdamOptimizerTest, SingleStepUpdateIsCorrect) {
  float lr = 0.1f, beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f;
  AdamOptimizer opt(lr, beta1, beta2, epsilon);

  Gaussians p = create_test_gaussians(1);
  Gradients g = create_test_gradients(p);

  Eigen::Vector3f initial_xyz = p.xyz[0];
  Eigen::Vector3f grad_xyz = g.xyz[0];

  // --- Manually calculate the expected update ---
  // Timestep t=1
  Eigen::Vector3f m1 = (1 - beta1) * grad_xyz;
  Eigen::Vector3f v1 = (1 - beta2) * grad_xyz.cwiseProduct(grad_xyz);

  float m_corr = 1.0f / (1.0f - beta1);
  float v_corr = 1.0f / (1.0f - beta2);

  Eigen::Vector3f m_hat = m1 * m_corr;
  Eigen::Vector3f v_hat = v1 * v_corr;

  Eigen::Vector3f expected_update = lr * m_hat.cwiseQuotient((v_hat.cwiseSqrt().array() + epsilon).matrix());
  Eigen::Vector3f expected_xyz = initial_xyz - expected_update;

  // --- Perform step and compare ---
  opt.step(p, g);

  ASSERT_TRUE(p.xyz[0].isApprox(expected_xyz, 1e-6));
}

// Test that all parameter types are updated after one step
TEST_F(AdamOptimizerTest, StepUpdatesAllParameters) {
  Gaussians initial_params = create_test_gaussians(1, 1);

  optimizer.step(params, grads);

  // Check that every parameter has changed from its initial value
  EXPECT_FALSE(params.xyz[0].isApprox(initial_params.xyz[0]));
  EXPECT_FALSE(params.rgb[0].isApprox(initial_params.rgb[0]));
  EXPECT_NE(params.opacity[0], initial_params.opacity[0]);
  EXPECT_FALSE(params.scale[0].isApprox(initial_params.scale[0]));
  EXPECT_FALSE(params.quaternion[0].isApprox(initial_params.quaternion[0]));
  ASSERT_TRUE(params.sh.has_value());
  EXPECT_FALSE((*params.sh)[0].isApprox((*initial_params.sh)[0]));
}

// Test the filter_states method
TEST_F(AdamOptimizerTest, FilterStates) {
  // Perform one step to initialize states
  optimizer.step(params, grads);

  // Filter out the middle element (index 1)
  std::vector<bool> mask = {true, false, true};
  optimizer.filter_states(mask);

  // Create new params and grads for the next step, matching the filtered size
  Gaussians filtered_params = create_test_gaussians(2);
  Gradients filtered_grads = create_test_gradients(filtered_params);

  // The next step should not throw an error, indicating state sizes are consistent
  ASSERT_NO_THROW(optimizer.step(filtered_params, filtered_grads));
}

// Test the filter_states method with an all-false mask
TEST_F(AdamOptimizerTest, FilterStatesAllFalse) {
  optimizer.step(params, grads);

  std::vector<bool> mask(params.size(), false);
  optimizer.filter_states(mask);

  Gaussians empty_params;
  Gradients empty_grads;
  ASSERT_NO_THROW(optimizer.step(empty_params, empty_grads));
}

// Test the append_states method
TEST_F(AdamOptimizerTest, AppendStates) {
  // Initialize states with 3 gaussians
  optimizer.step(params, grads);

  // Append states for 2 new gaussians
  size_t num_to_append = 2;
  optimizer.append_states(num_to_append);

  // Create new params and grads for the next step, matching the new total size
  Gaussians appended_params = create_test_gaussians(params.size() + num_to_append);
  Gradients appended_grads = create_test_gradients(appended_params);

  // The next step should not throw an error
  ASSERT_NO_THROW(optimizer.step(appended_params, appended_grads));
}

// Test a realistic sequence of step, filter, and append
TEST_F(AdamOptimizerTest, DynamicTrainingSimulation) {
  // 1. Initial state (3 gaussians)
  ASSERT_NO_THROW(optimizer.step(params, grads));

  // 2. Filter out one gaussian (now 2)
  std::vector<bool> filter_mask = {true, false, true};
  params.filter(filter_mask);
  optimizer.filter_states(filter_mask);
  ASSERT_EQ(params.size(), 2);

  // 3. Step with the filtered set
  grads = create_test_gradients(params);
  ASSERT_NO_THROW(optimizer.step(params, grads));

  // 4. Append three new gaussians (now 5)
  size_t num_to_append = 3;
  Gaussians new_gaussians = create_test_gaussians(num_to_append, 1);
  params.append(new_gaussians);
  optimizer.append_states(num_to_append);
  ASSERT_EQ(params.size(), 5);

  // 5. Step with the appended set
  grads = create_test_gradients(params);
  ASSERT_NO_THROW(optimizer.step(params, grads));
}

TEST_F(AdamOptimizerTest, UpgradeSHStates) {
  // 1. Initial state: Create and step with Gaussians that have NO SH coefficients.
  // This initializes the optimizer's states for all other parameters.
  size_t num_gaussians = 3;
  params = create_test_gaussians(num_gaussians, -1); // -1 indicates no SH
  grads = create_test_gradients(params);
  optimizer.step(params, grads);

  // 2. Test Initialization: Upgrade from no SH to SH band 1.
  // Band 1 has (1+1)^2 - 1 = 3 coefficients per color channel.
  const size_t band_1_total_coeffs = 3 * 3;

  // Call the function under test to initialize the optimizer's SH states.
  optimizer.upgrade_sh_states(num_gaussians, band_1_total_coeffs);

  // Create new Gaussians and Gradients that now include SH data.
  Gaussians params_sh1 = create_test_gaussians(num_gaussians, -1);
  params_sh1.sh.emplace(num_gaussians, Eigen::VectorXf::Random(band_1_total_coeffs));
  Gradients grads_sh1 = create_test_gradients(params_sh1);

  // A subsequent step should succeed without throwing an error.
  ASSERT_NO_THROW(optimizer.step(params_sh1, grads_sh1));

  // 3. Test Upgrading: Upgrade from SH band 1 to SH band 2.
  // Band 2 has (2+1)^2 - 1 = 8 coefficients per color channel.
  const size_t band_2_total_coeffs = 8 * 3;

  // Call the function under test again to upgrade the existing SH states.
  optimizer.upgrade_sh_states(num_gaussians, band_2_total_coeffs);

  // Create new Gaussians and Gradients with the higher-degree SH data.
  Gaussians params_sh2 = create_test_gaussians(num_gaussians, -1);
  params_sh2.sh.emplace(num_gaussians, Eigen::VectorXf::Random(band_2_total_coeffs));
  Gradients grads_sh2 = create_test_gradients(params_sh2);

  // This step should also succeed, confirming the states were resized correctly.
  ASSERT_NO_THROW(optimizer.step(params_sh2, grads_sh2));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
