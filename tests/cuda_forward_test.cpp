#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include "gsplat/cuda_forward.hpp" // For kernel function declarations

// Macro for checking CUDA API calls for errors.
#define CUDA_CHECK(err)                                                                                                \
  do {                                                                                                                 \
    cudaError_t err_ = (err);                                                                                          \
    if (err_ != cudaSuccess) {                                                                                         \
      fprintf(stderr, "CUDA error at %s:%d, error code: %d (%s)\n", __FILE__, __LINE__, err_,                          \
              cudaGetErrorString(err_));                                                                               \
      FAIL();                                                                                                          \
    }                                                                                                                  \
  } while (0)

// Test fixture for tests that require CUDA device allocations.
// Using a fixture helps manage setup and teardown of device memory.
class CudaKernelTest : public ::testing::Test {
protected:
  // Per-test-suite set-up.
  // Called before the first test in this test suite.
  static void SetUpTestSuite() {
    // Ensure a CUDA device is available
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    ASSERT_GT(deviceCount, 0);
  }
};

// Test case for the camera_intrinsic_projection kernel.
TEST_F(CudaKernelTest, CameraProjection) {
  const int N = 4; // Number of points

  // Host-side data
  // K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
  const std::vector<float> h_K = {100.0f, 0.0f, 50.0f, 0.0f, 120.0f, 60.0f, 0.0f, 0.0f, 1.0f};
  const float fx = h_K[0], cx = h_K[2], fy = h_K[4], cy = h_K[5];

  const std::vector<float> h_xyz = {
      1.0f,  1.0f,  2.0f, // Point 0
      2.0f,  -3.0f, 5.0f, // Point 1
      0.0f,  0.0f,  1.0f, // Point 2
      -4.0f, 2.0f,  10.0f // Point 3
  };
  std::vector<float> h_uv(N * 2);

  // Device-side data pointers
  float *d_K, *d_xyz, *d_uv;

  // Allocate memory on the device
  CUDA_CHECK(cudaMalloc(&d_K, h_K.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_xyz, h_xyz.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_uv, h_uv.size() * sizeof(float)));

  // Copy input data from host to device
  CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), h_K.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), h_xyz.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Launch the kernel
  camera_intrinsic_projection(d_xyz, d_K, N, d_uv);
  CUDA_CHECK(cudaDeviceSynchronize()); // Wait for the kernel to finish

  // Copy result data from device to host
  CUDA_CHECK(cudaMemcpy(h_uv.data(), d_uv, h_uv.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // Calculate expected results on the host
  std::vector<float> expected_uv(N * 2);
  for (int i = 0; i < N; ++i) {
    const float x = h_xyz[i * 3 + 0];
    const float y = h_xyz[i * 3 + 1];
    const float z = h_xyz[i * 3 + 2];
    expected_uv[i * 2 + 0] = fx * x / z + cx;
    expected_uv[i * 2 + 1] = fy * y / z + cy;
  }

  // Compare results
  for (int i = 0; i < N * 2; ++i) {
    ASSERT_NEAR(h_uv[i], expected_uv[i], 1e-5);
  }

  // Free device memory
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_xyz));
  CUDA_CHECK(cudaFree(d_uv));
}

// Test case for the cull_gaussians kernel.
TEST_F(CudaKernelTest, GaussianCulling) {
  const int N = 7; // Number of points
  // Culling parameters
  const float near_thresh = 1.0f;
  const float far_thresh = 10.0f;
  const int padding = 10;
  const int width = 1920;
  const int height = 1080;

  // Host-side data
  // Some points are inside, some are outside, some are on the edge.
  const std::vector<float> h_xyz = {
      0.0f, 0.0f, 5.0f,  // Case 1: In view, should be kept.
      0.0f, 0.0f, 0.5f,  // Case 2: Too close (z < near), but in frustum, should be kept.
      0.0f, 0.0f, 12.0f, // Case 3: Too far (z > far), but in frustum, should be kept.
      0.0f, 0.0f, 5.0f,  // Case 4: In left padding, should be kept.
      0.0f, 0.0f, 5.0f,  // Case 5: In right padding, should be kept.
      0.0f, 0.0f, 12.0f, // Case 6: Too far AND outside frustum, should be CULLED.
      0.0f, 0.0f, 0.5f   // Case 7: Too close AND outside frustum, should be CULLED.
  };
  const std::vector<float> h_uv = {
      960.0f,  540.0f, // Case 1
      960.0f,  540.0f, // Case 2
      960.0f,  540.0f, // Case 3
      -5.0f,   540.0f, // Case 4
      1925.0f, 540.0f, // Case 5
      -11.0f,  540.0f, // Case 6
      960.0f,  1091.0f // Case 7
  };
  std::vector<uint8_t> h_mask(N);

  // Device-side data pointers
  float *d_xyz, *d_uv;
  bool *d_mask;

  // Allocate memory on the device
  CUDA_CHECK(cudaMalloc(&d_xyz, h_xyz.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_uv, h_uv.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_mask, h_mask.size() * sizeof(bool)));

  // Copy input data from host to device
  CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), h_xyz.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_uv, h_uv.data(), h_uv.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Launch the kernel
  cull_gaussians(d_uv, d_xyz, N, near_thresh, far_thresh, padding, width, height, d_mask);
  CUDA_CHECK(cudaDeviceSynchronize()); // Wait for the kernel to finish

  // Copy result data from device to host
  CUDA_CHECK(cudaMemcpy(h_mask.data(), d_mask, h_mask.size() * sizeof(bool), cudaMemcpyDeviceToHost));

  // Expected mask values. The mask is true if the point should be CULLED.
  // Logic from kernel: mask = !(z_ok || uv_ok)
  const std::vector<bool> expected_mask = {
      false, // Case 1: Keep
      false, // Case 2: Keep
      false, // Case 3: Keep
      false, // Case 4: Keep
      false, // Case 5: Keep
      true,  // Case 6: CULL
      true   // Case 7: CULL
  };

  // Compare results
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(h_mask[i], expected_mask[i]) << "Mismatch at index " << i;
  }

  // Free device memory
  CUDA_CHECK(cudaFree(d_xyz));
  CUDA_CHECK(cudaFree(d_uv));
  CUDA_CHECK(cudaFree(d_mask));
}
