#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include "gsplat/adaptive_density.hpp"

// Helper macro for checking CUDA calls
#define CUDA_CHECK(call)                                                                                               \
  {                                                                                                                    \
    cudaError_t err = call;                                                                                            \
    ASSERT_EQ(err, cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err);                                          \
  }

// A test fixture for managing common setup and teardown for GPU resources
class AdaptiveDensityTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    // Ensure a CUDA device is available
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    ASSERT_GT(deviceCount, 0);
  }

  // Pointers to device memory for Gaussian attributes
  float *d_xyz = nullptr, *d_rgb = nullptr, *d_sh = nullptr, *d_opacity = nullptr;
  float *d_scale = nullptr, *d_quaternion = nullptr;
  float *d_uv_grad_accum = nullptr, *d_xyz_grad_accum = nullptr;
  int *d_grad_accum_count = nullptr;
  bool *d_mask = nullptr;

  // Pointers to device memory for optimizer moment vectors
  float *d_m_xyz = nullptr, *d_v_xyz = nullptr, *d_m_rgb = nullptr, *d_v_rgb = nullptr;
  float *d_m_sh = nullptr, *d_v_sh = nullptr, *d_m_opacity = nullptr, *d_v_opacity = nullptr;
  float *d_m_scale = nullptr, *d_v_scale = nullptr, *d_m_quaternion = nullptr, *d_v_quaternion = nullptr;

  // Parameters
  int N;
  int max_gaussians;
  int num_sh_coef;

  // Clean up all allocated device memory
  void TearDown() override {
    // Free Gaussian attributes
    if (d_xyz)
      CUDA_CHECK(cudaFree(d_xyz));
    if (d_rgb)
      CUDA_CHECK(cudaFree(d_rgb));
    if (d_sh)
      CUDA_CHECK(cudaFree(d_sh));
    if (d_opacity)
      CUDA_CHECK(cudaFree(d_opacity));
    if (d_scale)
      CUDA_CHECK(cudaFree(d_scale));
    if (d_quaternion)
      CUDA_CHECK(cudaFree(d_quaternion));
    if (d_uv_grad_accum)
      CUDA_CHECK(cudaFree(d_uv_grad_accum));
    if (d_xyz_grad_accum)
      CUDA_CHECK(cudaFree(d_xyz_grad_accum));
    if (d_grad_accum_count)
      CUDA_CHECK(cudaFree(d_grad_accum_count));
    if (d_mask)
      CUDA_CHECK(cudaFree(d_mask));

    // Free optimizer moments
    if (d_m_xyz)
      CUDA_CHECK(cudaFree(d_m_xyz));
    if (d_v_xyz)
      CUDA_CHECK(cudaFree(d_v_xyz));
    if (d_m_rgb)
      CUDA_CHECK(cudaFree(d_m_rgb));
    if (d_v_rgb)
      CUDA_CHECK(cudaFree(d_v_rgb));
    if (d_m_sh)
      CUDA_CHECK(cudaFree(d_m_sh));
    if (d_v_sh)
      CUDA_CHECK(cudaFree(d_v_sh));
    if (d_m_opacity)
      CUDA_CHECK(cudaFree(d_m_opacity));
    if (d_v_opacity)
      CUDA_CHECK(cudaFree(d_v_opacity));
    if (d_m_scale)
      CUDA_CHECK(cudaFree(d_m_scale));
    if (d_v_scale)
      CUDA_CHECK(cudaFree(d_v_scale));
    if (d_m_quaternion)
      CUDA_CHECK(cudaFree(d_m_quaternion));
    if (d_v_quaternion)
      CUDA_CHECK(cudaFree(d_v_quaternion));
  }

  // Helper to initialize host and device data for a test
  void InitializeData(int n_gaussians, int max_g, int sh_c, const std::vector<float> &h_opacity,
                      const std::vector<float> &h_scale, const std::vector<float> &h_uv_grad,
                      const std::vector<int> &h_grad_count) {
    N = n_gaussians;
    max_gaussians = max_g;
    num_sh_coef = sh_c;

    // Allocate device memory for Gaussian attributes
    CUDA_CHECK(cudaMalloc(&d_xyz, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rgb, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_opacity, max_gaussians * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scale, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_quaternion, max_gaussians * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_uv_grad_accum, max_gaussians * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xyz_grad_accum, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_accum_count, max_gaussians * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mask, max_gaussians * sizeof(bool)));

    // Allocate device memory for optimizer moments
    CUDA_CHECK(cudaMalloc(&d_m_xyz, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_xyz, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_rgb, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_rgb, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_opacity, max_gaussians * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_opacity, max_gaussians * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_scale, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_scale, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_quaternion, max_gaussians * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_quaternion, max_gaussians * 4 * sizeof(float)));

    if (num_sh_coef > 0) {
      CUDA_CHECK(cudaMalloc(&d_sh, max_gaussians * num_sh_coef * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_m_sh, max_gaussians * num_sh_coef * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_v_sh, max_gaussians * num_sh_coef * sizeof(float)));
    }

    // Create dummy host data for attributes
    std::vector<float> h_xyz(N * 3, 1.0f);
    std::vector<float> h_rgb(N * 3, 0.5f);
    std::vector<float> h_quat(N * 4, 0.0f);
    h_quat[0] = 1.0f; // Identity quaternion
    std::vector<float> h_xyz_grad(N * 3, 0.01f);

    // Copy attribute data to device
    CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_quaternion, h_quat.data(), N * 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xyz_grad_accum, h_xyz_grad.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_opacity, h_opacity.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale, h_scale.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uv_grad_accum, h_uv_grad.data(), N * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_accum_count, h_grad_count.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize optimizer moments to zero on device
    CUDA_CHECK(cudaMemset(d_m_xyz, 0, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_xyz, 0, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_rgb, 0, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_rgb, 0, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_opacity, 0, max_gaussians * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_opacity, 0, max_gaussians * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_scale, 0, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_scale, 0, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_quaternion, 0, max_gaussians * 4 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_quaternion, 0, max_gaussians * 4 * sizeof(float)));
    if (num_sh_coef > 0) {
      CUDA_CHECK(cudaMemset(d_m_sh, 0, max_gaussians * num_sh_coef * sizeof(float)));
      CUDA_CHECK(cudaMemset(d_v_sh, 0, max_gaussians * num_sh_coef * sizeof(float)));
    }
  }
};

// Test to ensure nothing happens when all operations are disabled
TEST_F(AdaptiveDensityTest, NoOperation) {
  InitializeData(4, 8, 0, {0.5f, 0.5f, 0.5f, 0.5f},                                        // opacity
                 {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, // scale
                 {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},                         // uv_grad
                 {1, 1, 1, 1}                                                              // grad_count
  );

  adaptive_density(N, 0, num_sh_coef, false, 0, 0, 0.0f, false, 0.0f, 0.0f, max_gaussians, false, false, false, 0.1f,
                   0.001f, 2, 1.6f, d_uv_grad_accum, d_grad_accum_count, d_scale, d_mask, d_xyz_grad_accum, d_xyz,
                   d_rgb, d_sh, d_opacity, d_quaternion, d_m_xyz, d_v_xyz, d_m_rgb, d_v_rgb, d_m_opacity, d_v_opacity,
                   d_m_scale, d_v_scale, d_m_quaternion, d_v_quaternion, d_m_sh, d_v_sh);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<char> h_mask(max_gaussians);
  CUDA_CHECK(cudaMemcpy(h_mask.data(), d_mask, max_gaussians * sizeof(bool), cudaMemcpyDeviceToHost));

  // The mask should be all false since it's initialized to false and no operations are enabled.
  for (int i = 0; i < N; ++i) {
    ASSERT_FALSE(h_mask[i]);
  }
}

// Test the deletion logic based on opacity, gradient norm, and accumulation count
TEST_F(AdaptiveDensityTest, DeletionLogic) {
  // Kernel uses logit opacity. op_param_threshold = logf(0.1) - (1.0 - 0.1) = -3.202
  const float high_op = 0.9f;       // stays
  const float low_op_logit = -4.0f; // gets deleted

  InitializeData(4, 8, 0, {high_op, low_op_logit, high_op, high_op},           // opacity
                 {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, // scale
                 {
                     1.0f,
                     1.0f, // G0: Keep
                     1.0f,
                     1.0f, // G1: Delete (low opacity)
                     0.0f,
                     0.0f, // G2: Delete (zero grad)
                     1.0f,
                     1.0f,    // G3: Delete (zero accum count)
                 },           // uv_grad
                 {1, 1, 1, 0} // grad_count
  );

  adaptive_density(N, 0, num_sh_coef, false, 0, 0, 0.0f, false, 0.0f, 0.0f, max_gaussians, true, false, false, 0.1f,
                   0.0f, 2, 1.6f, d_uv_grad_accum, d_grad_accum_count, d_scale, d_mask, d_xyz_grad_accum, d_xyz, d_rgb,
                   d_sh, d_opacity, d_quaternion, d_m_xyz, d_v_xyz, d_m_rgb, d_v_rgb, d_m_opacity, d_v_opacity,
                   d_m_scale, d_v_scale, d_m_quaternion, d_v_quaternion, d_m_sh, d_v_sh);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<char> h_mask(max_gaussians);
  CUDA_CHECK(cudaMemcpy(h_mask.data(), d_mask, max_gaussians * sizeof(bool), cudaMemcpyDeviceToHost));

  EXPECT_TRUE(h_mask[0]);  // Kept
  EXPECT_FALSE(h_mask[1]); // Deleted (low opacity)
  EXPECT_FALSE(h_mask[2]); // Deleted (zero grad)
  EXPECT_FALSE(h_mask[3]); // Deleted (zero accum count)
}

// Test the cloning logic for small Gaussians with high gradients
TEST_F(AdaptiveDensityTest, CloningLogic) {
  const float uv_split_val = 0.5f;
  const float clone_scale_threshold = 0.1f;

  InitializeData(2, 4, 0, {0.9f, 0.9f}, // opacity
                 {
                     logf(0.05f),
                     logf(0.05f),
                     logf(0.05f), // G0: small scale, should clone
                     logf(0.2f),
                     logf(0.2f),
                     logf(0.2f), // G1: large scale, should not clone
                 },              // scale (log scale)
                 {
                     1.0f,
                     1.0f, // G0: High grad norm > uv_split_val
                     1.0f,
                     1.0f, // G1: High grad norm > uv_split_val
                 },        // uv_grad
                 {1, 1}    // grad_count
  );

  adaptive_density(N, 0, num_sh_coef, false, 0, 0, uv_split_val, false, 0.0f, 0.0f, max_gaussians, true, true, false,
                   0.01f, clone_scale_threshold, 2, 1.6f, d_uv_grad_accum, d_grad_accum_count, d_scale, d_mask,
                   d_xyz_grad_accum, d_xyz, d_rgb, d_sh, d_opacity, d_quaternion, d_m_xyz, d_v_xyz, d_m_rgb, d_v_rgb,
                   d_m_opacity, d_v_opacity, d_m_scale, d_v_scale, d_m_quaternion, d_v_quaternion, d_m_sh, d_v_sh);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<char> h_mask(max_gaussians);
  std::vector<float> h_xyz(max_gaussians * 3);
  std::vector<float> h_opacity(max_gaussians);
  CUDA_CHECK(cudaMemcpy(h_mask.data(), d_mask, max_gaussians * sizeof(bool), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_xyz.data(), d_xyz, max_gaussians * 3 * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_opacity.data(), d_opacity, max_gaussians * sizeof(float), cudaMemcpyDeviceToHost));

  // G0 should be kept, and a new Gaussian should be created
  EXPECT_TRUE(h_mask[0]);
  // G1 should be kept, but not cloned
  EXPECT_TRUE(h_mask[1]);
  // A new Gaussian should be created at index N=2 from cloning G0
  EXPECT_TRUE(h_mask[2]);
  // No other gaussians should be created
  EXPECT_FALSE(h_mask[3]);

  // Verify the cloned Gaussian's properties
  EXPECT_EQ(h_opacity[2], h_opacity[0]);
  EXPECT_NEAR(h_xyz[2 * 3 + 0], 1.0f - 0.01f * 0.01f, 1e-6); // original_xyz - xyz_grad * 0.01
}

// Test the splitting logic for large Gaussians
TEST_F(AdaptiveDensityTest, SplittingLogic) {
  const float uv_split_val = 0.5f;
  const float clone_scale_threshold = 0.1f;
  const int num_split_samples = 2;
  const float split_scale_factor = 1.6f;

  InitializeData(2, 8, 0, {0.9f, 0.9f}, // opacity
                 {
                     logf(0.2f),
                     logf(0.2f),
                     logf(0.2f), // G0: large scale, high grad -> should split
                     logf(0.05f),
                     logf(0.05f),
                     logf(0.05f), // G1: small scale, high grad -> should not split
                 },               // scale (log scale)
                 {
                     1.0f,
                     1.0f, // G0: High grad norm > uv_split_val
                     1.0f,
                     1.0f, // G1: High grad norm > uv_split_val
                 },        // uv_grad
                 {1, 1}    // grad_count
  );

  adaptive_density(N, 0, num_sh_coef, false, 0, 0, uv_split_val, false, 0.0f, 0.0f, max_gaussians, true, false, true,
                   0.01f, clone_scale_threshold, num_split_samples, split_scale_factor, d_uv_grad_accum,
                   d_grad_accum_count, d_scale, d_mask, d_xyz_grad_accum, d_xyz, d_rgb, d_sh, d_opacity, d_quaternion,
                   d_m_xyz, d_v_xyz, d_m_rgb, d_v_rgb, d_m_opacity, d_v_opacity, d_m_scale, d_v_scale, d_m_quaternion,
                   d_v_quaternion, d_m_sh, d_v_sh);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<char> h_mask(max_gaussians);
  std::vector<float> h_scale(max_gaussians * 3);
  CUDA_CHECK(cudaMemcpy(h_mask.data(), d_mask, max_gaussians * sizeof(bool), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_scale.data(), d_scale, max_gaussians * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  // G0 is split, so it should be marked for deletion
  EXPECT_FALSE(h_mask[0]);
  // G1 is not split, so it is kept
  EXPECT_TRUE(h_mask[1]);

  // Two new Gaussians should be created at index N=2 and N+1=3 from splitting G0
  EXPECT_TRUE(h_mask[2]);
  EXPECT_TRUE(h_mask[3]);
  EXPECT_FALSE(h_mask[4]);

  // Verify the new scales
  float expected_new_scale = logf(0.2f / split_scale_factor);
  EXPECT_NEAR(h_scale[2 * 3 + 0], expected_new_scale, 1e-6);
  EXPECT_NEAR(h_scale[3 * 3 + 0], expected_new_scale, 1e-6);
}

// Test a mixed scenario with deletion, cloning, and splitting in one call
TEST_F(AdaptiveDensityTest, CombinedOperations) {
  const float uv_split_val = 0.5f;
  const float clone_scale_threshold = 0.1f;
  const float delete_op_threshold = 0.1f;
  const int num_split_samples = 2;

  InitializeData(3, 10, 0,
                 {
                     -4.0f, // G0: Delete (low opacity)
                     0.9f,  // G1: Clone
                     0.9f,  // G2: Split
                 },         // opacity (logit)
                 {
                     logf(0.05f),
                     logf(0.05f),
                     logf(0.05f),
                     logf(0.05f),
                     logf(0.05f),
                     logf(0.05f), // G1: small scale
                     logf(0.2f),
                     logf(0.2f),
                     logf(0.2f), // G2: large scale
                 },              // scale (log)
                 {
                     1.0f,
                     1.0f, // G0
                     1.0f,
                     1.0f, // G1 (high grad)
                     1.0f,
                     1.0f, // G2 (high grad)
                 },        // uv_grad
                 {1, 1, 1} // grad_count
  );

  adaptive_density(N, 0, num_sh_coef, false, 0, 0, uv_split_val, false, 0.0f, 0.0f, max_gaussians, true, true, true,
                   delete_op_threshold, clone_scale_threshold, num_split_samples, 1.6f, d_uv_grad_accum,
                   d_grad_accum_count, d_scale, d_mask, d_xyz_grad_accum, d_xyz, d_rgb, d_sh, d_opacity, d_quaternion,
                   d_m_xyz, d_v_xyz, d_m_rgb, d_v_rgb, d_m_opacity, d_v_opacity, d_m_scale, d_v_scale, d_m_quaternion,
                   d_v_quaternion, d_m_sh, d_v_sh);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<char> h_mask(max_gaussians);
  CUDA_CHECK(cudaMemcpy(h_mask.data(), d_mask, max_gaussians * sizeof(bool), cudaMemcpyDeviceToHost));

  // G0 deleted (via keep_test returning false)
  EXPECT_FALSE(h_mask[0]);
  // G1 kept and cloned
  EXPECT_TRUE(h_mask[1]);
  // G2 split (original deleted)
  EXPECT_FALSE(h_mask[2]);

  // Check new Gaussians. G1 clone goes first.
  // New gaussians start writing at index N=3
  // G1 clone writes 1 element to index 3
  EXPECT_TRUE(h_mask[3]);
  // G2 split writes 2 elements to index 4, 5
  EXPECT_TRUE(h_mask[4]);
  EXPECT_TRUE(h_mask[5]);
  // Rest should be false
  EXPECT_FALSE(h_mask[6]);
}
