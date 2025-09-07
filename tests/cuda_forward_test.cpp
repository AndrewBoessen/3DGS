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

// Test case for the get_sorted_gaussian_list function.
// This function operates in two passes:
// 1. (sorted_gaussians == nullptr): Calculates the total number of splats (gaussian-tile overlaps)
//    and returns the required size for the sort_keys buffer.
// 2. (sorted_gaussians != nullptr): Populates the buffers with the sorted list of gaussian indices
//    and the start/end indices for each tile.
TEST_F(CudaKernelTest, GetSortedGaussianList) {
  const int N = 3; // Number of Gaussians
  const int width = 64;
  const int height = 64;
  const int tile_dim = 16;
  const int n_tiles_x = width / tile_dim;
  const int n_tiles_y = height / tile_dim;
  const int num_tiles = n_tiles_x * n_tiles_y;
  const float mh_dist = 3.0f; // Mahalanobis distance for 99% confidence

  // Host-side input data
  // Gaussian 0: Small, entirely within tile (1,1). z=10.
  // Gaussian 1: Larger, on the border of tile (1,1) and (2,1). z=20.
  // Gaussian 2: Small, entirely within tile (2,2). z=5.
  const std::vector<float> h_uvs = {
      24.0f, 24.0f, // G0 -> center of tile (1,1)
      32.0f, 24.0f, // G1 -> border of tiles (1,1) and (2,1)
      40.0f, 40.0f  // G2 -> center of tile (2,2)
  };
  const std::vector<float> h_xyz = {
      0.0f, 0.0f, 10.0f, // G0
      0.0f, 0.0f, 20.0f, // G1
      0.0f, 0.0f, 5.0f   // G2
  };
  // Conic parameters a,b,c. For a circle, b=0, a=c. Radius ~ mh_dist * sqrt(a).
  // G0 & G2 radius = 4 => 3*sqrt(a)=4 => a=16/9 ~= 1.78
  // G1 radius = 6 => 3*sqrt(a)=6 => a=36/9 = 4
  const std::vector<float> h_conic = {
      1.78f, 0.0f, 1.78f, // G0
      4.0f,  0.0f, 4.0f,  // G1
      1.78f, 0.0f, 1.78f  // G2
  };

  // Device-side pointers
  float *d_uvs, *d_xyz, *d_conic;
  int *d_sorted_gaussians, *d_splat_boundaries;

  // Allocate and copy inputs to device
  CUDA_CHECK(cudaMalloc(&d_uvs, h_uvs.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_xyz, h_xyz.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_conic, h_conic.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_uvs, h_uvs.data(), h_uvs.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), h_xyz.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_conic, h_conic.data(), h_conic.size() * sizeof(float), cudaMemcpyHostToDevice));

  // --- PASS 1: Get required buffer size ---
  size_t sorted_gaussian_bytes = 0;
  get_sorted_gaussian_list(d_uvs, d_xyz, d_conic, n_tiles_x, n_tiles_y, mh_dist, N, sorted_gaussian_bytes, nullptr,
                           nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Expected splats:
  // G0 -> tile (1,1) [idx 5]
  // G1 -> tiles (1,1) [idx 5] and (2,1) [idx 6]
  // G2 -> tile (2,2) [idx 10]
  // Total = 4 splats. The size is for the `double` sort keys.
  const int num_splats = 4;
  ASSERT_EQ(sorted_gaussian_bytes, num_splats * sizeof(int));

  // --- PASS 2: Execute with allocated buffers ---
  CUDA_CHECK(cudaMalloc(&d_sorted_gaussians, sorted_gaussian_bytes));
  CUDA_CHECK(cudaMalloc(&d_splat_boundaries, (num_tiles + 1) * sizeof(int)));

  get_sorted_gaussian_list(d_uvs, d_xyz, d_conic, n_tiles_x, n_tiles_y, mh_dist, N, sorted_gaussian_bytes,
                           d_sorted_gaussians, d_splat_boundaries);
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- Verification ---
  std::vector<int> h_sorted_gaussians(num_splats);
  std::vector<int> h_splat_boundaries(num_tiles + 1);
  CUDA_CHECK(
      cudaMemcpy(h_sorted_gaussians.data(), d_sorted_gaussians, num_splats * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h_splat_boundaries.data(), d_splat_boundaries, (num_tiles + 1) * sizeof(int), cudaMemcpyDeviceToHost));

  // Expected sorting: by tile index, then by z-depth.
  // Tile 5: G0 (z=10), G1 (z=20) -> sorted order: [0, 1]
  // Tile 6: G1 (z=20) -> sorted order: [1]
  // Tile 10: G2 (z=5) -> sorted order: [2]
  // Final list of gaussian indices: [0, 1, 1, 2]
  const std::vector<int> expected_sorted_gaussians = {0, 1, 1, 2};
  for (int i = 0; i < num_splats; ++i) {
    EXPECT_EQ(h_sorted_gaussians[i], expected_sorted_gaussians[i]) << "Mismatch at sorted gaussian index " << i;
  }

  // Expected tile boundaries: `arr[i]` is start, `arr[i+1]` is end for tile `i`.
  // Tile 5: [0, 2)
  // Tile 6: [2, 3)
  // Tile 10: [3, 4)
  std::vector<int> expected_splat_boundaries(num_tiles + 1, 0);
  expected_splat_boundaries[5] = 0;
  expected_splat_boundaries[6] = 2; // End of tile 5, start of tile 6
  expected_splat_boundaries[7] = 3; // End of tile 6
  // Tiles 7,8,9 are empty. The start idx remains the end of the previous tile.
  expected_splat_boundaries[8] = 3;
  expected_splat_boundaries[9] = 3;
  expected_splat_boundaries[10] = 3; // Start of tile 10
  expected_splat_boundaries[11] = 4; // End of tile 10

  // The kernel only writes at boundaries, so empty intermediate indices might not be updated.
  // We only check the critical boundary points.
  EXPECT_EQ(h_splat_boundaries[5], 0);
  EXPECT_EQ(h_splat_boundaries[6], 2);
  EXPECT_EQ(h_splat_boundaries[7], 3);
  EXPECT_EQ(h_splat_boundaries[10], 3);
  EXPECT_EQ(h_splat_boundaries[11], 4);
  // Check that an empty tile's range is valid (start >= end)
  EXPECT_GE(h_splat_boundaries[9], h_splat_boundaries[8]);

  // --- Cleanup ---
  CUDA_CHECK(cudaFree(d_uvs));
  CUDA_CHECK(cudaFree(d_xyz));
  CUDA_CHECK(cudaFree(d_conic));
  CUDA_CHECK(cudaFree(d_sorted_gaussians));
  CUDA_CHECK(cudaFree(d_splat_boundaries));
}

// Test case for the precompute_spherical_harmonics function.
TEST_F(CudaKernelTest, PrecomputeSphericalHarmonics) {
  // 1. Setup test parameters
  const int N = 2;
  const int l_max = 1;
  const int n_coeffs = (l_max + 1) * (l_max + 1); // (1+1)^2 = 4 coefficients

  // 2. Host-side data
  // Input xyz (normalized direction vectors)
  const std::vector<float> h_xyz = {
      0.0f, 0.0f, 1.0f, // Point 0: View direction along Z-axis
      1.0f, 0.0f, 0.0f  // Point 1: View direction along X-axis
  };

  // Input SH coefficients. Layout is (N, n_coeffs, 3).
  const std::vector<float> h_sh_coefficients = {
      // Point 0 coeffs (R, G, B for each of the 4 basis functions)
      0.5f, -0.2f, 0.8f, // l=0, m=0
      0.1f, 0.1f, 0.1f,  // l=1, m=-1
      0.2f, 0.2f, 0.2f,  // l=1, m=0
      0.3f, 0.3f, 0.3f,  // l=1, m=1
      // Point 1 coeffs
      0.1f, 0.5f, 0.9f, // l=0, m=0
      0.2f, 0.6f, 0.0f, // l=1, m=-1
      0.3f, 0.7f, 0.1f, // l=1, m=0
      0.4f, 0.8f, 0.2f  // l=1, m=1
  };
  std::vector<float> h_rgb(N * 3);

  // 3. Device-side data setup
  float *d_xyz, *d_sh_coefficients, *d_rgb;
  CUDA_CHECK(cudaMalloc(&d_xyz, h_xyz.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sh_coefficients, h_sh_coefficients.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_rgb, h_rgb.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), h_xyz.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sh_coefficients, h_sh_coefficients.data(), h_sh_coefficients.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // 4. Call the function to be tested
  precompute_spherical_harmonics(d_xyz, d_sh_coefficients, l_max, N, d_rgb);
  CUDA_CHECK(cudaDeviceSynchronize());

  // 5. Copy results back to host
  CUDA_CHECK(cudaMemcpy(h_rgb.data(), d_rgb, h_rgb.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // 6. Calculate expected results on the host
  // Real SH basis functions for l_max=1:
  // Y_0_0(x,y,z) = 0.28209
  // Y_1_-1(x,y,z) = 0.48860 * y
  // Y_1_0(x,y,z)  = 0.48860 * z
  // Y_1_1(x,y,z)  = 0.48860 * x

  // For Point 0 (x=0, y=0, z=1):
  // SH basis values = {0.28209, 0.0, 0.48860, 0.0}
  float sum_r0 = (0.5f * 0.28209f) + (0.2f * 0.48860f);  // 0.238765
  float sum_g0 = (-0.2f * 0.28209f) + (0.2f * 0.48860f); // 0.041302
  float sum_b0 = (0.8f * 0.28209f) + (0.2f * 0.48860f);  // 0.323392

  // For Point 1 (x=1, y=0, z=0):
  // SH basis values = {0.28209, 0.0, 0.0, 0.48860}
  float sum_r1 = (0.1f * 0.28209f) + (0.4f * 0.48860f); // 0.223649
  float sum_g1 = (0.5f * 0.28209f) + (0.8f * 0.48860f); // 0.531925
  float sum_b1 = (0.9f * 0.28209f) + (0.2f * 0.48860f); // 0.351601

  auto sigmoid = [](float x) { return 1.0f / (1.0f + expf(-x)); };
  const std::vector<float> expected_rgb = {sigmoid(sum_r0), sigmoid(sum_g0), sigmoid(sum_b0),
                                           sigmoid(sum_r1), sigmoid(sum_g1), sigmoid(sum_b1)};

  // 7. Compare results
  for (size_t i = 0; i < h_rgb.size(); ++i) {
    ASSERT_NEAR(h_rgb[i], expected_rgb[i], 1e-4);
  }

  // 8. Cleanup
  CUDA_CHECK(cudaFree(d_xyz));
  CUDA_CHECK(cudaFree(d_sh_coefficients));
  CUDA_CHECK(cudaFree(d_rgb));
}

// Test case for the render_image function with multiple Gaussians.
// This test renders three gaussians onto a single tile and verifies the output.
TEST_F(CudaKernelTest, RenderImageMultipleGaussians) {
  // 1. Setup test parameters
  const int width = 16;
  const int height = 16;
  const int N = 3; // Number of Gaussians

  // 2. Host-side input data
  // Three Gaussians with different properties
  const std::vector<float> h_uv = {
      7.5f,  7.5f, // Gaussian 1 (center)
      3.5f,  3.5f, // Gaussian 2 (top-left)
      11.5f, 11.5f // Gaussian 3 (bottom-right)
  };
  const std::vector<float> h_opacity = {0.5f, 0.6f, 0.4f};
  const std::vector<float> h_rgb = {
      1.0f, 0.8f, 0.4f, // Orange-ish
      0.4f, 0.8f, 1.0f, // Blue-ish
      0.8f, 1.0f, 0.4f  // Green-ish
  };
  const std::vector<float> h_conic = {
      1.0f, 0.0f,  1.0f, // Circle
      2.0f, 0.5f,  2.0f, // Ellipse 1
      1.5f, -0.5f, 1.5f  // Ellipse 2
  };

  // The sorted list includes all three gaussians for the single tile
  const std::vector<int> h_sorted_splats = {0, 1, 2};
  // The splat range for tile 0 is from index 0 to 3
  const std::vector<int> h_splat_range_by_tile = {0, 3};

  // 3. Host-side output buffers
  std::vector<float> h_image(width * height * 3);
  std::vector<float> h_weight_per_pixel(width * height);

  // 4. Device-side data setup
  float *d_uv, *d_opacity, *d_conic, *d_rgb, *d_weight_per_pixel, *d_image;
  int *d_sorted_splats, *d_splat_range_by_tile;

  CUDA_CHECK(cudaMalloc(&d_uv, h_uv.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_opacity, h_opacity.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_conic, h_conic.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_rgb, h_rgb.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sorted_splats, h_sorted_splats.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_splat_range_by_tile, h_splat_range_by_tile.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_weight_per_pixel, h_weight_per_pixel.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_image, h_image.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_uv, h_uv.data(), h_uv.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_opacity, h_opacity.data(), h_opacity.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_conic, h_conic.data(), h_conic.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb.data(), h_rgb.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sorted_splats, h_sorted_splats.data(), h_sorted_splats.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_splat_range_by_tile, h_splat_range_by_tile.data(), h_splat_range_by_tile.size() * sizeof(int),
                        cudaMemcpyHostToDevice));

  // 5. Call the function to be tested
  render_image(d_uv, d_opacity, d_conic, d_rgb, d_sorted_splats, d_splat_range_by_tile, width, height,
               d_weight_per_pixel, d_image);
  CUDA_CHECK(cudaDeviceSynchronize());

  // 6. Copy results back to host
  CUDA_CHECK(cudaMemcpy(h_image.data(), d_image, h_image.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // 7. Calculate expected results on the host.
  // This helper function replicates the kernel's alpha blending logic for a single pixel.
  auto calculate_expected_color = [&](float u_pixel, float v_pixel) {
    // Start with a white background and zero accumulated alpha
    float r = 1.0f, g = 1.0f, b = 1.0f;
    float alpha_accum = 0.0f;

    // Iterate through gaussians in reverse order (front to back)
    for (int i = 0; i < N; ++i) {
      const int gaussian_idx = h_sorted_splats[i];
      const float u_mean = h_uv[gaussian_idx * 2];
      const float v_mean = h_uv[gaussian_idx * 2 + 1];

      const float u_diff = u_pixel - u_mean;
      const float v_diff = v_pixel - v_mean;

      const float a = h_conic[gaussian_idx * 3 + 0] + 0.25f;
      const float b_c = h_conic[gaussian_idx * 3 + 1] + 0.5f;
      const float c = h_conic[gaussian_idx * 3 + 2] + 0.25f;

      const float det = a * c - b_c * b_c;
      const float mh_sq = (c * u_diff * u_diff - (b_c + b_c) * u_diff * v_diff + a * v_diff * v_diff) / det;

      float alpha = 0.0f;
      if (mh_sq > 0.0f) {
        alpha = h_opacity[gaussian_idx] * expf(-0.5f * mh_sq);
      }

      // Kernel's blending logic
      const float T_i = 1.0f - alpha_accum;
      alpha = alpha * T_i;

      r += (h_rgb[gaussian_idx * 3 + 0] - r) * alpha;
      g += (h_rgb[gaussian_idx * 3 + 1] - g) * alpha;
      b += (h_rgb[gaussian_idx * 3 + 2] - b) * alpha;

      alpha_accum += alpha;
    }
    return std::vector<float>{r, g, b};
  };

  // Check the central pixel: (7, 7), which is close to the first gaussian
  int idx_center = (7 * width + 7) * 3;
  std::vector<float> expected_center = calculate_expected_color(7.0f, 7.0f);
  ASSERT_NEAR(h_image[idx_center + 0], expected_center[0], 1e-2);
  ASSERT_NEAR(h_image[idx_center + 1], expected_center[1], 1e-2);
  ASSERT_NEAR(h_image[idx_center + 2], expected_center[2], 1e-2);

  // Check a pixel far from all gaussians: (0, 0)
  // Its color should be nearly pure white background.
  int idx_corner = (0 * width + 0) * 3;
  std::vector<float> expected_corner = calculate_expected_color(0.0f, 0.0f);
  ASSERT_NEAR(h_image[idx_corner + 0], expected_corner[0], 1e-2);
  ASSERT_NEAR(h_image[idx_corner + 1], expected_corner[1], 1e-2);
  ASSERT_NEAR(h_image[idx_corner + 2], expected_corner[2], 1e-2);
  ASSERT_NEAR(h_image[idx_corner + 0], 1.0f, 1e-2); // Check against white
  ASSERT_NEAR(h_image[idx_corner + 1], 1.0f, 1e-2);
  ASSERT_NEAR(h_image[idx_corner + 2], 1.0f, 1e-2);

  // 8. Cleanup
  CUDA_CHECK(cudaFree(d_uv));
  CUDA_CHECK(cudaFree(d_opacity));
  CUDA_CHECK(cudaFree(d_conic));
  CUDA_CHECK(cudaFree(d_rgb));
  CUDA_CHECK(cudaFree(d_sorted_splats));
  CUDA_CHECK(cudaFree(d_splat_range_by_tile));
  CUDA_CHECK(cudaFree(d_weight_per_pixel));
  CUDA_CHECK(cudaFree(d_image));
}
