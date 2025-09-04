#include <cmath>
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

// Test case for the compute_sigma function.
// This test verifies the calculation of the covariance matrix Sigma = (R*S) * (R*S)^T,
// where R is a rotation matrix and S is a diagonal scaling matrix.
// The output Sigma matrix is stored in column-major order.
TEST_F(CudaKernelTest, ComputeSigma) {
  const int N = 2;

  // Host-side input data
  const std::vector<float> h_quaternion = {// Case 1: Identity rotation
                                           1.0f, 0.0f, 0.0f, 0.0f,
                                           // Case 2: 90-degree rotation around Z-axis
                                           sqrtf(0.5f), 0.0f, 0.0f, sqrtf(0.5f)};

  const std::vector<float> h_scale = {// Case 1: Scales for identity rotation
                                      logf(2.0f), logf(3.0f), logf(4.0f),
                                      // Case 2: Scales for rotated gaussian
                                      logf(1.0f), logf(2.0f), logf(3.0f)};

  std::vector<float> h_sigma(N * 9); // Each sigma is a 3x3 matrix

  // Device-side data pointers
  float *d_quaternion, *d_scale, *d_sigma;

  // Allocate memory on the device
  CUDA_CHECK(cudaMalloc(&d_quaternion, h_quaternion.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_scale, h_scale.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sigma, h_sigma.size() * sizeof(float)));

  // Copy input data from host to device
  CUDA_CHECK(
      cudaMemcpy(d_quaternion, h_quaternion.data(), h_quaternion.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale, h_scale.data(), h_scale.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Launch the function to be tested
  compute_sigma(d_quaternion, d_scale, N, d_sigma);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result from device to host
  CUDA_CHECK(cudaMemcpy(h_sigma.data(), d_sigma, h_sigma.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // Expected results calculated on the host
  // The output sigma is in COLUMN-MAJOR order.
  const std::vector<float> expected_sigma = {// Case 1: R=I, S=diag(2,3,4). Sigma = diag(4,9,16)
                                             // Column 1   Column 2   Column 3
                                             4.0f, 0.0f, 0.0f, 0.0f, 9.0f, 0.0f, 0.0f, 0.0f, 16.0f,

                                             // Case 2: R=RotZ(90), S=diag(1,2,3). Sigma = diag(4,1,9) after rotation.
                                             // Column 1   Column 2   Column 3
                                             4.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 9.0f};

  // Compare results
  for (size_t i = 0; i < h_sigma.size(); ++i) {
    ASSERT_NEAR(h_sigma[i], expected_sigma[i], 1e-5);
  }

  // Free device memory
  CUDA_CHECK(cudaFree(d_quaternion));
  CUDA_CHECK(cudaFree(d_scale));
  CUDA_CHECK(cudaFree(d_sigma));
}

// Test case for the camera_intrinsic_projection kernel.
TEST_F(CudaKernelTest, CameraIntrinsicProjection) {
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

// Test case for the camera_extrinsic_projection function.
TEST_F(CudaKernelTest, CameraExtrinsicProjection) {
  const int N = 3; // Number of points

  // Host-side data
  // Extrinsic matrix T = [R|t] is 3x4.
  // R is identity, t = [10, 20, 30].
  const std::vector<float> h_T = {
      1.0f, 0.0f, 0.0f, 10.0f, // Row 1
      0.0f, 1.0f, 0.0f, 20.0f, // Row 2
      0.0f, 0.0f, 1.0f, 30.0f  // Row 3
  };

  // World coordinates (x, y, z)
  const std::vector<float> h_xyz_w = {
      1.0f,  2.0f, 3.0f,  // Point 0
      -5.0f, 4.0f, -1.0f, // Point 1
      0.0f,  0.0f, 0.0f   // Point 2
  };

  // This will store the output camera coordinates.
  std::vector<float> h_xyz_c(N * 3);

  // Device-side data pointers
  float *d_T, *d_xyz_w, *d_xyz_c;

  // Allocate memory on the device
  CUDA_CHECK(cudaMalloc(&d_T, h_T.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_xyz_w, h_xyz_w.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_xyz_c, h_xyz_c.size() * sizeof(float)));

  // Copy input data from host to device
  CUDA_CHECK(cudaMemcpy(d_T, h_T.data(), h_T.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_xyz_w, h_xyz_w.data(), h_xyz_w.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Launch the function (which wraps a CUBLAS call)
  camera_extrinsic_projection(d_xyz_w, d_T, N, d_xyz_c);
  CUDA_CHECK(cudaDeviceSynchronize()); // Wait for the kernel to finish

  // Copy result data from device to host
  CUDA_CHECK(cudaMemcpy(h_xyz_c.data(), d_xyz_c, h_xyz_c.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // Calculate expected results on the host
  // xyz_c = R * xyz_w + t
  std::vector<float> expected_xyz_c(N * 3);
  for (int i = 0; i < N; ++i) {
    const float x_w = h_xyz_w[i * 3 + 0];
    const float y_w = h_xyz_w[i * 3 + 1];
    const float z_w = h_xyz_w[i * 3 + 2];
    // Since R is identity, this simplifies to x_c = x_w + t_x, etc.
    expected_xyz_c[i * 3 + 0] = x_w + h_T[3];  // t_x
    expected_xyz_c[i * 3 + 1] = y_w + h_T[7];  // t_y
    expected_xyz_c[i * 3 + 2] = z_w + h_T[11]; // t_z
  }

  // Compare results
  for (int i = 0; i < N * 3; ++i) {
    ASSERT_NEAR(h_xyz_c[i], expected_xyz_c[i], 1e-5);
  }

  // Free device memory
  CUDA_CHECK(cudaFree(d_T));
  CUDA_CHECK(cudaFree(d_xyz_w));
  CUDA_CHECK(cudaFree(d_xyz_c));
}

// Test case for the compute_conic function.
// This test verifies Jacobian calculation and then conic calculation.
TEST_F(CudaKernelTest, ComputeConic) {
  const int N = 1; // Test with a single Gaussian

  // Host-side input data
  const std::vector<float> h_xyz = {1.0f, 2.0f, 5.0f}; // Camera-space coordinates
  const std::vector<float> h_K = {100.0f, 0.0f, 50.0f, 0.0f, 120.0f, 60.0f, 0.0f, 0.0f, 1.0f}; // Intrinsics
  const std::vector<float> h_sigma = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}; // 3x3 Identity covariance
  const std::vector<float> h_T = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}; // Identity extrinsics

  // Host-side output buffers
  std::vector<float> h_J(N * 6);
  std::vector<float> h_conic(N * 3);

  // Device-side pointers
  float *d_xyz, *d_K, *d_sigma, *d_T, *d_J, *d_conic;

  // Allocate memory on the device
  CUDA_CHECK(cudaMalloc(&d_xyz, h_xyz.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_K, h_K.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sigma, h_sigma.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_T, h_T.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_J, h_J.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_conic, h_conic.size() * sizeof(float)));

  // Copy input data from host to device
  CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), h_xyz.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), h_K.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sigma, h_sigma.data(), h_sigma.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_T, h_T.data(), h_T.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Launch the function to be tested
  compute_conic(d_xyz, d_K, d_sigma, d_T, N, d_J, d_conic);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result from device to host
  CUDA_CHECK(cudaMemcpy(h_conic.data(), d_conic, h_conic.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // --- Calculate expected results on the host for verification ---
  const float x = h_xyz[0], y = h_xyz[1], z = h_xyz[2];
  const float fx = h_K[0], fy = h_K[4];

  // 1. Expected Jacobian J
  const float j00 = fx / z;
  const float j02 = -fx * x / (z * z);
  const float j11 = fy / z;
  const float j12 = -fy * y / (z * z);

  // 2. W is identity because T is identity
  // 3. M = J @ W = J
  // 4. V = Sigma @ M^T = Identity @ J^T = J^T
  // 5. Conic = M @ V = J @ J^T
  const float c00 = j00 * j00 + 0.0f * 0.0f + j02 * j02;
  const float c01 = j00 * 0.0f + 0.0f * j11 + j02 * j12;
  const float c11 = 0.0f * 0.0f + j11 * j11 + j12 * j12;

  const std::vector<float> expected_conic = {c00, c01, c11};

  // Compare results
  for (size_t i = 0; i < h_conic.size(); ++i) {
    ASSERT_NEAR(h_conic[i], expected_conic[i], 1e-5);
  }

  // Free device memory
  CUDA_CHECK(cudaFree(d_xyz));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_sigma));
  CUDA_CHECK(cudaFree(d_T));
  CUDA_CHECK(cudaFree(d_J));
  CUDA_CHECK(cudaFree(d_conic));
}
