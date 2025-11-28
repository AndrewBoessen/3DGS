#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include "gsplat_cuda/adaptive_density.cuh"

// Helper macro for checking CUDA calls
#define CUDA_CHECK(err)                                                                                                \
  do {                                                                                                                 \
    cudaError_t err_ = (err);                                                                                          \
    if (err_ != cudaSuccess) {                                                                                         \
      fprintf(stderr, "CUDA error at %s:%d, error code: %d (%s)\n", __FILE__, __LINE__, err_,                          \
              cudaGetErrorString(err_));                                                                               \
      FAIL();                                                                                                          \
    }                                                                                                                  \
  } while (0)

// A test fixture for managing common setup and teardown for GPU resources
class AdaptiveDensityTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    // Ensure a CUDA device is available
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    ASSERT_GT(deviceCount, 0);
  }

  // Pointers to device memory for Gaussian attributes (INPUTS)
  float *d_xyz = nullptr, *d_rgb = nullptr, *d_sh = nullptr, *d_opacity = nullptr;
  float *d_scale = nullptr, *d_quaternion = nullptr;
  float *d_uv_grad_accum = nullptr;
  int *d_grad_accum_count = nullptr;
  bool *d_mask = nullptr;
  int *d_write_ids = nullptr;

  // Pointers to device memory for optimizer moment vectors (not used by these kernels)
  float *d_m_xyz = nullptr, *d_v_xyz = nullptr, *d_m_rgb = nullptr, *d_v_rgb = nullptr;
  float *d_m_sh = nullptr, *d_v_sh = nullptr, *d_m_opacity = nullptr, *d_v_opacity = nullptr;
  float *d_m_scale = nullptr, *d_v_scale = nullptr, *d_m_quaternion = nullptr, *d_v_quaternion = nullptr;

  // Pointers to device memory for kernel OUTPUTS
  float *d_xyz_out = nullptr, *d_rgb_out = nullptr, *d_sh_out = nullptr, *d_opacity_out = nullptr;
  float *d_scale_out = nullptr, *d_quaternion_out = nullptr;

  // Parameters
  int N;
  int max_gaussians;
  int num_sh_coef;

  // Clean up all allocated device memory
  void TearDown() override {
    // Free Gaussian attributes (inputs)
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
    if (d_grad_accum_count)
      CUDA_CHECK(cudaFree(d_grad_accum_count));
    if (d_mask)
      CUDA_CHECK(cudaFree(d_mask));
    if (d_write_ids)
      CUDA_CHECK(cudaFree(d_write_ids));

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

    // Free output buffers
    if (d_xyz_out)
      CUDA_CHECK(cudaFree(d_xyz_out));
    if (d_rgb_out)
      CUDA_CHECK(cudaFree(d_rgb_out));
    if (d_sh_out)
      CUDA_CHECK(cudaFree(d_sh_out));
    if (d_opacity_out)
      CUDA_CHECK(cudaFree(d_opacity_out));
    if (d_scale_out)
      CUDA_CHECK(cudaFree(d_scale_out));
    if (d_quaternion_out)
      CUDA_CHECK(cudaFree(d_quaternion_out));
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
    CUDA_CHECK(cudaMalloc(&d_grad_accum_count, max_gaussians * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mask, max_gaussians * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_write_ids, max_gaussians * sizeof(int)));

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

    // Allocate output buffers
    CUDA_CHECK(cudaMalloc(&d_xyz_out, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rgb_out, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_opacity_out, max_gaussians * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scale_out, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_quaternion_out, max_gaussians * 4 * sizeof(float)));

    if (num_sh_coef > 0) {
      CUDA_CHECK(cudaMalloc(&d_sh, max_gaussians * num_sh_coef * 3 * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_m_sh, max_gaussians * num_sh_coef * 3 * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_v_sh, max_gaussians * num_sh_coef * 3 * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_sh_out, max_gaussians * num_sh_coef * 3 * sizeof(float)));
    }

    // Create dummy host data for attributes
    std::vector<float> h_xyz(N * 3, 1.0f);
    std::vector<float> h_rgb(N * 3, 0.5f);
    std::vector<float> h_quat(N * 4, 0.0f);
    for (int i = 0; i < N; ++i) {
      h_quat[i * 4 + 0] = 1.0f; // Identity quaternion w
      h_quat[i * 4 + 3] = 0.0f; // Identity quaternion z (kernel expects w,x,y,z)
    }

    // Copy attribute data to device
    CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_quaternion, h_quat.data(), N * 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_opacity, h_opacity.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale, h_scale.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_uv_grad_accum, h_uv_grad.data(), N * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_accum_count, h_grad_count.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize optimizer moments to zero on device
    CUDA_CHECK(cudaMemset(d_m_xyz, 0, max_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_xyz, 0, max_gaussians * 3 * sizeof(float)));
    // ... (rest of memsets)
  }
};

// Test the clone_gaussians kernel directly
TEST_F(AdaptiveDensityTest, CloneGaussiansTest) {
  N = 2;
  max_gaussians = 4;
  num_sh_coef = 0;

  // Setup initial data
  InitializeData(N, max_gaussians, num_sh_coef, {0.9f, 0.9f}, // opacity
                 {1.f, 1.f, 1.f, 1.f, 1.f, 1.f},              // scale
                 {1.f, 1.f, 1.f, 1.f},                        // uv_grad
                 {10, 1}                                      // grad_count
  );

  // Set specific input data for this test
  std::vector<float> h_xyz_in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<char> h_mask = {true, false}; // Clone G0, not G1
  std::vector<int> h_write_ids = {0, 1};    // Exclusive scan of h_mask
  std::vector<float> h_opacity_in = {0.8f, 0.7f};

  CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz_in.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(), N * sizeof(bool), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_write_ids, h_write_ids.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_opacity, h_opacity_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // Call the kernel
  clone_gaussians(N, num_sh_coef, d_mask, d_write_ids, d_xyz, d_rgb, d_opacity, d_scale, d_quaternion, d_sh, d_xyz_out,
                  d_rgb_out, d_opacity_out, d_scale_out, d_quaternion_out, d_sh_out, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy results back
  std::vector<float> h_xyz_out(max_gaussians * 3);
  std::vector<float> h_opacity_out(max_gaussians);
  CUDA_CHECK(cudaMemcpy(h_xyz_out.data(), d_xyz_out, max_gaussians * 3 * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_opacity_out.data(), d_opacity_out, max_gaussians * sizeof(float), cudaMemcpyDeviceToHost));

  // Verify G0 (index 0) was cloned.
  // G0 has write_id = 0, so write_base = 0 * 2 = 0. Writes to indices 0 and 1.
  // G0 grad_accum = 100 / 10 = 10.0
  // G0 clone_factor = 10.0 * 0.01 = 0.1
  // G0 xyz_in = {1.0, 2.0, 3.0}
  EXPECT_NEAR(h_xyz_out[0], 1.0f, 1e-6);
  EXPECT_NEAR(h_xyz_out[1], 2.0f, 1e-6);
  EXPECT_NEAR(h_xyz_out[2], 3.0f, 1e-6);
  // Original xyz (j=1) = {1.0, 2.0, 3.0}
  EXPECT_NEAR(h_xyz_out[3], 1.0f, 1e-6);
  EXPECT_NEAR(h_xyz_out[4], 2.0f, 1e-6);
  EXPECT_NEAR(h_xyz_out[5], 3.0f, 1e-6);

  // Check opacity was copied
  EXPECT_EQ(h_opacity_out[0], h_opacity_in[0]);
  EXPECT_EQ(h_opacity_out[1], h_opacity_in[0]);

  // G1 (index 1) was not processed, so h_xyz_out[6]... are uninitialized (or 0)
  // and h_opacity_out[2]... are uninitialized.
}

// Test the split_gaussians kernel directly
TEST_F(AdaptiveDensityTest, SplitGaussiansTest) {
  N = 2;
  max_gaussians = 4;
  num_sh_coef = 0;
  const float scale_factor = 1.6f;

  // Setup initial data
  InitializeData(N, max_gaussians, num_sh_coef, {0.9f, 0.9f}, // opacity
                 {1.f, 1.f, 1.f, 1.f, 1.f, 1.f},              // scale
                 {1.f, 1.f, 1.f, 1.f},                        // uv_grad
                 {10, 1}                                      // grad_count
  );

  // Set specific input data for this test
  std::vector<float> h_scale_in = {logf(2.0f), logf(2.0f), logf(2.0f), logf(0.1f), logf(0.1f), logf(0.1f)};
  std::vector<char> h_mask = {true, false}; // Split G0, not G1
  std::vector<int> h_write_ids = {0, 1};    // Exclusive scan of h_mask
  std::vector<float> h_opacity_in = {0.8f, 0.7f};

  CUDA_CHECK(cudaMemcpy(d_scale, h_scale_in.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(), N * sizeof(bool), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_write_ids, h_write_ids.data(), N * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_opacity, h_opacity_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // Call the kernel
  split_gaussians(N, scale_factor, num_sh_coef, d_mask, d_write_ids, d_xyz, d_rgb, d_opacity, d_scale, d_quaternion,
                  d_sh, d_xyz_out, d_rgb_out, d_opacity_out, d_scale_out, d_quaternion_out, d_sh_out, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy results back
  std::vector<float> h_scale_out(max_gaussians * 3);
  std::vector<float> h_opacity_out(max_gaussians);
  CUDA_CHECK(cudaMemcpy(h_scale_out.data(), d_scale_out, max_gaussians * 3 * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_opacity_out.data(), d_opacity_out, max_gaussians * sizeof(float), cudaMemcpyDeviceToHost));

  // Verify G0 (index 0) was split.
  // G0 has write_id = 0, so write_base = 0 * 2 = 0. Writes to indices 0 and 1.
  // G0 scale_in = logf(2.0). exp_scale = 2.0.
  // New scale = logf(exp_scale / scale_factor) = logf(2.0 / 1.6)
  float expected_scale = logf(2.0f / scale_factor);

  // Check new scales for the two new Gaussians
  EXPECT_NEAR(h_scale_out[0], expected_scale, 1e-6);
  EXPECT_NEAR(h_scale_out[1], expected_scale, 1e-6);
  EXPECT_NEAR(h_scale_out[2], expected_scale, 1e-6);

  EXPECT_NEAR(h_scale_out[3], expected_scale, 1e-6);
  EXPECT_NEAR(h_scale_out[4], expected_scale, 1e-6);
  EXPECT_NEAR(h_scale_out[5], expected_scale, 1e-6);

  // Check opacity was copied
  EXPECT_EQ(h_opacity_out[0], h_opacity_in[0]);
  EXPECT_EQ(h_opacity_out[1], h_opacity_in[0]);

  // Note: We cannot deterministically test the new xyz positions
  // because the kernel uses curand_normal with a seed based on time(NULL).
  // Testing the scale and opacity copy is sufficient to validate the kernel's execution.
}
