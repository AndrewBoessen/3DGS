#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include "gsplat_cuda/cuda_forward.cuh"

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

  std::vector<float> h_sigma(N * 6); // Each sigma is a symmetric 3x3 matrix (stored as 6 floats)

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
  // The output sigma is in stored as [xx, xy, xz, yy, yz, zz]
  const std::vector<float> expected_sigma = {// Case 1: R=I, S=diag(2,3,4). Sigma = diag(4,9,16)
                                             4.0f, 0.0f, 0.0f, 9.0f, 0.0f, 16.0f,

                                             // Case 2: R=RotZ(90), S=diag(1,2,3). Sigma = diag(4,1,9) after rotation.
                                             4.0f, 0.0f, 0.0f, 1.0f, 0.0f, 9.0f};

  // Compare results
  for (size_t i = 0; i < h_sigma.size(); ++i) {
    ASSERT_NEAR(h_sigma[i], expected_sigma[i], 1e-4);
  }

  // Free device memory
  CUDA_CHECK(cudaFree(d_quaternion));
  CUDA_CHECK(cudaFree(d_scale));
  CUDA_CHECK(cudaFree(d_sigma));
}

// Test case for the project_to_screen kernel.
TEST_F(CudaKernelTest, ProjectToScreen) {
  const int N = 4; // Number of points
  const int width = 1920;
  const int height = 1080;

  const std::vector<float> h_proj = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};

  const std::vector<float> h_xyz = {
      1.0f,  1.0f,  2.0f, // Point 0: x/z = 0.5, y/z = 0.5 -> uv = (0.75*w, 0.75*h)
      2.0f,  -3.0f, 5.0f, // Point 1: x/z = 0.4, y/z = -0.6 -> uv = (0.7*w, 0.2*h)
      0.0f,  0.0f,  1.0f, // Point 2: 0, 0 -> uv = (0.5*w, 0.5*h)
      -4.0f, 2.0f,  10.0f // Point 3: -0.4, 0.2 -> uv = (0.3*w, 0.6*h)
  };
  std::vector<float> h_uv(N * 2);

  // Device-side data pointers
  float *d_proj, *d_xyz, *d_uv;

  // Allocate memory on the device
  CUDA_CHECK(cudaMalloc(&d_proj, h_proj.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_xyz, h_xyz.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_uv, h_uv.size() * sizeof(float)));

  // Copy input data from host to device
  CUDA_CHECK(cudaMemcpy(d_proj, h_proj.data(), h_proj.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), h_xyz.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Launch the kernel
  project_to_screen(d_xyz, d_proj, N, width, height, d_uv);
  CUDA_CHECK(cudaDeviceSynchronize()); // Wait for the kernel to finish

  // Copy result data from device to host
  CUDA_CHECK(cudaMemcpy(h_uv.data(), d_uv, h_uv.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // Calculate expected results on the host
  std::vector<float> expected_uv(N * 2);
  for (int i = 0; i < N; ++i) {
    const float x = h_xyz[i * 3 + 0];
    const float y = h_xyz[i * 3 + 1];
    const float z = h_xyz[i * 3 + 2];
    // With our custom Proj:
    // x_clip = x
    // y_clip = y
    // w_clip = z
    // x_ndc = x / z
    // y_ndc = y / z
    // u = (x_ndc * 0.5 + 0.5) * width
    // v = (y_ndc * 0.5 + 0.5) * height

    expected_uv[i * 2 + 0] = (x / z * 0.5f + 0.5f) * width;
    expected_uv[i * 2 + 1] = (y / z * 0.5f + 0.5f) * height;
  }

  // Compare results
  for (int i = 0; i < N * 2; ++i) {
    ASSERT_NEAR(h_uv[i], expected_uv[i], 1e-3);
  }

  // Free device memory
  CUDA_CHECK(cudaFree(d_proj));
  CUDA_CHECK(cudaFree(d_xyz));
  CUDA_CHECK(cudaFree(d_uv));
}

// Test case for the cull_gaussians kernel.
TEST_F(CudaKernelTest, GaussianCulling) {
  const int N = 7; // Number of points
  // Culling parameters
  const float near_thresh = 1.0f;
  const int padding = 10;
  const int width = 1920;
  const int height = 1080;

  // Host-side data
  // Some points are inside, some are outside, some are on the edge.
  const std::vector<float> h_xyz = {
      0.0f, 0.0f, 5.0f,  // Case 1: In view, should be kept.
      0.0f, 0.0f, 0.5f,  // Case 2: Too close (z < near), but in frustum, should be CULLED.
      0.0f, 0.0f, 12.0f, // Case 3: In frustum, should be kept.
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
  cull_gaussians(d_uv, d_xyz, N, near_thresh, padding, width, height, d_mask);
  CUDA_CHECK(cudaDeviceSynchronize()); // Wait for the kernel to finish

  // Copy result data from device to host
  CUDA_CHECK(cudaMemcpy(h_mask.data(), d_mask, h_mask.size() * sizeof(bool), cudaMemcpyDeviceToHost));

  // Expected mask values. The mask is true if the point should be CULLED.
  // Logic from kernel: mask = !(z_ok || uv_ok)
  const std::vector<bool> expected_mask = {
      true,  // Case 1: Keep
      false, // Case 2: CULL
      true,  // Case 3: Keep
      true,  // Case 4: Keep
      true,  // Case 5: Keep
      false, // Case 6: CULL
      false  // Case 7: CULL
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

// Test case for the compute_camera_space_points function.
TEST_F(CudaKernelTest, ComputeCameraSpacePoints) {
  const int N = 3; // Number of points

  // Host-side data
  // View matrix V = [R|t] is 4x4.
  // R is identity, t = [10, 20, 30].
  // V =
  // 1 0 0 10
  // 0 1 0 20
  // 0 0 1 30
  // 0 0 0 1
  const std::vector<float> h_view = {
      1.0f, 0.0f, 0.0f, 10.0f, // Row 1
      0.0f, 1.0f, 0.0f, 20.0f, // Row 2
      0.0f, 0.0f, 1.0f, 30.0f, // Row 3
      0.0f, 0.0f, 0.0f, 1.0f   // Row 4
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
  float *d_view, *d_xyz_w, *d_xyz_c;

  // Allocate memory on the device
  CUDA_CHECK(cudaMalloc(&d_view, h_view.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_xyz_w, h_xyz_w.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_xyz_c, h_xyz_c.size() * sizeof(float)));

  // Copy input data from host to device
  CUDA_CHECK(cudaMemcpy(d_view, h_view.data(), h_view.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_xyz_w, h_xyz_w.data(), h_xyz_w.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Launch the function
  compute_camera_space_points(d_xyz_w, d_view, N, d_xyz_c);
  CUDA_CHECK(cudaDeviceSynchronize()); // Wait for the kernel to finish

  // Copy result data from device to host
  CUDA_CHECK(cudaMemcpy(h_xyz_c.data(), d_xyz_c, h_xyz_c.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // Calculate expected results on the host
  // xyz_c = V * xyz_w
  std::vector<float> expected_xyz_c(N * 3);
  for (int i = 0; i < N; ++i) {
    const float x_w = h_xyz_w[i * 3 + 0];
    const float y_w = h_xyz_w[i * 3 + 1];
    const float z_w = h_xyz_w[i * 3 + 2];
    // Since R is identity, this simplifies to x_c = x_w + t_x, etc.
    expected_xyz_c[i * 3 + 0] = x_w + h_view[3];  // t_x
    expected_xyz_c[i * 3 + 1] = y_w + h_view[7];  // t_y
    expected_xyz_c[i * 3 + 2] = z_w + h_view[11]; // t_z
  }

  // Compare results
  for (int i = 0; i < N * 3; ++i) {
    ASSERT_NEAR(h_xyz_c[i], expected_xyz_c[i], 1e-5);
  }

  // Free device memory
  CUDA_CHECK(cudaFree(d_view));
  CUDA_CHECK(cudaFree(d_xyz_w));
  CUDA_CHECK(cudaFree(d_xyz_c));
}

// Test case for the compute_conic function.
// This test verifies Jacobian calculation and then conic calculation.
TEST_F(CudaKernelTest, ComputeConic) {
  const int N = 1; // Test with a single Gaussian

  // Host-side input data
  const std::vector<float> h_xyz = {1.0f, 2.0f, 5.0f}; // Camera-space coordinates
  // Proj matrix (4x4)
  // Use simple identity-like projection for easy Jacobian verification
  // P =
  // 1 0 0 0
  // 0 1 0 0
  // 0 0 0 1
  // 0 0 1 0
  // This means x_proj = x/z, y_proj = y/z
  const std::vector<float> h_proj = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};

  const std::vector<float> h_sigma = {1.0f, 0.0f, 0.0f,
                                      1.0f, 0.0f, 1.0f}; // 3x3 Identity covariance (xx, xy, xz, yy, yz, zz)
  const std::vector<float> h_view = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}; // Identity view

  // Host-side output buffers
  std::vector<float> h_J(N * 6);
  std::vector<float> h_conic(N * 3);
  std::vector<float4> h_radius(N);

  // Device-side pointers
  float *d_xyz, *d_proj, *d_sigma, *d_view, *d_J, *d_conic;
  float4 *d_radius;

  // Allocate memory on the device
  CUDA_CHECK(cudaMalloc(&d_xyz, h_xyz.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_proj, h_proj.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sigma, h_sigma.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_view, h_view.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_J, h_J.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_conic, h_conic.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_radius, N * sizeof(float4)));

  // Copy input data from host to device
  CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), h_xyz.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_proj, h_proj.data(), h_proj.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sigma, h_sigma.data(), h_sigma.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_view, h_view.data(), h_view.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Launch the function to be tested
  compute_conic(d_xyz, d_view, d_sigma, 1.0f, 1.0f, 1.0f, 1.0f, 3.0f, N, d_J, d_conic, d_radius);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result from device to host
  CUDA_CHECK(cudaMemcpy(h_conic.data(), d_conic, h_conic.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_radius.data(), d_radius, h_radius.size() * sizeof(float4), cudaMemcpyDeviceToHost));

  // --- Calculate expected results on the host for verification ---
  const float x = h_xyz[0], y = h_xyz[1], z = h_xyz[2];
  // With our simple Proj:
  // x_ndc = x / z
  // y_ndc = y / z
  // J = d(uv)/d(xyz)
  // u = x/z * W/2 + W/2
  // v = y/z * H/2 + H/2
  // But wait, the kernel computes J for NDC coordinates (or screen? check kernel)
  // The kernel computes J = d(x_proj, y_proj) / d(x, y, z)
  // J = [ 1/z, 0, -x/z^2 ]
  //     [ 0, 1/z, -y/z^2 ]
  // (Assuming p_proj.x = x/z, p_proj.y = y/z)

  const float j00 = 1.0f / z;
  const float j02 = -x / (z * z);
  const float j11 = 1.0f / z;
  const float j12 = -y / (z * z);

  // 2. W is identity because View is identity
  // 3. M = J @ W = J
  // 4. V = Sigma @ M^T = Identity @ J^T = J^T
  // 5. Covariance = M @ V = J @ J^T
  const float cov00 = j00 * j00 + 0.0f * 0.0f + j02 * j02 + 0.3f;
  const float cov01 = j00 * 0.0f + 0.0f * j11 + j02 * j12;
  const float cov11 = 0.0f * 0.0f + j11 * j11 + j12 * j12 + 0.3f;

  // 6. Conic = Inverse(Covariance)
  const float det = cov00 * cov11 - cov01 * cov01;
  const float inv_det = 1.0f / det;
  const float expected_c00 = cov11 * inv_det;
  const float expected_c01 = -cov01 * inv_det;
  const float expected_c11 = cov00 * inv_det;

  const std::vector<float> expected_conic = {expected_c00, expected_c01, expected_c11};

  // Compare results
  for (size_t i = 0; i < h_conic.size(); ++i) {
    ASSERT_NEAR(h_conic[i], expected_conic[i], 1e-5);
  }

  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(h_radius[i].x, 3.0f, 1e-5);
    EXPECT_NEAR(h_radius[i].y, 1.0f, 1e-5);
    EXPECT_NEAR(h_radius[i].z, sqrt(0.8), 1e-5);
    EXPECT_NEAR(h_radius[i].w, sqrt(0.2), 1e-5);
  }

  // Free device memory
  CUDA_CHECK(cudaFree(d_xyz));
  CUDA_CHECK(cudaFree(d_proj));
  CUDA_CHECK(cudaFree(d_sigma));
  CUDA_CHECK(cudaFree(d_view));
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
  // G0 & G2 radius = 4
  // G1 radius = 6
  const std::vector<float4> h_radius = {
      {4.0f, 4.0f, 0.f, 1.f}, // G0
      {4.0f, 4.0f, 0.f, 1.f}, // G1
      {6.0f, 6.0f, 0.f, 1.f}  // G2
  };

  // Device-side pointers
  float *d_uvs, *d_xyz;
  float4 *d_radius;
  int *d_sorted_gaussians, *d_splat_boundaries;

  // Allocate and copy inputs to device
  CUDA_CHECK(cudaMalloc(&d_uvs, h_uvs.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_xyz, h_xyz.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_radius, h_radius.size() * sizeof(float4)));
  CUDA_CHECK(cudaMemcpy(d_uvs, h_uvs.data(), h_uvs.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), h_xyz.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_radius, h_radius.data(), h_radius.size() * sizeof(float4), cudaMemcpyHostToDevice));

  // --- PASS 1: Get required buffer size ---
  size_t sorted_gaussian_bytes = 0;
  get_sorted_gaussian_list(d_uvs, d_xyz, d_radius, n_tiles_x, n_tiles_y, N, sorted_gaussian_bytes, nullptr, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Expected splats:
  // G0 -> tile (1,1) [idx 5]
  // G1 -> tiles (1,1) [idx 5] and (2,1) [idx 6]
  // G2 -> tile (2,2) [idx 10]
  // Total Pairs = 3 gaussians * 16 tiles to search.
  const int num_splats = 4;
  const int num_splats_size = 3 * 4 * 4;
  ASSERT_EQ(sorted_gaussian_bytes, num_splats_size);

  // --- PASS 2: Execute with allocated buffers ---
  CUDA_CHECK(cudaMalloc(&d_sorted_gaussians, sorted_gaussian_bytes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_splat_boundaries, (num_tiles + 1) * sizeof(int)));

  get_sorted_gaussian_list(d_uvs, d_xyz, d_radius, n_tiles_x, n_tiles_y, N, sorted_gaussian_bytes, d_sorted_gaussians,
                           d_splat_boundaries);
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
  CUDA_CHECK(cudaFree(d_radius));
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

  const std::vector<float> h_band_0 = {
      0.5f, -0.2f, 0.8f, // l=0, m=0
      0.1f, 0.5f,  0.9f, // l=0, m=0
  };

  // Input SH coefficients. Layout is (N, n_coeffs, 3).
  const std::vector<float> h_sh_coefficients = {
      // Point 0 coeffs (R, G, B for each of the 4 basis functions)
      0.1f, 0.1f, 0.1f, // l=1, m=-1
      0.2f, 0.2f, 0.2f, // l=1, m=0
      0.3f, 0.3f, 0.3f, // l=1, m=1
      // Point 1 coeffs
      0.2f, 0.6f, 0.0f, // l=1, m=-1
      0.3f, 0.7f, 0.1f, // l=1, m=0
      0.4f, 0.8f, 0.2f  // l=1, m=1
  };
  std::vector<float> h_rgb(N * 3);

  // 3. Device-side data setup
  float *d_xyz, *d_sh_coefficients, *d_band_0, *d_rgb;

  // Allocate dummy campos at origin
  float3 campos = {0.0f, 0.0f, 0.0f};

  CUDA_CHECK(cudaMalloc(&d_xyz, h_xyz.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sh_coefficients, h_sh_coefficients.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_band_0, h_band_0.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_rgb, h_rgb.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), h_xyz.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sh_coefficients, h_sh_coefficients.data(), h_sh_coefficients.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_band_0, h_band_0.data(), h_band_0.size() * sizeof(float), cudaMemcpyHostToDevice));

  // 4. Call the function to be tested
  // 4. Call the function to be tested
  precompute_spherical_harmonics(d_xyz, d_sh_coefficients, d_band_0, campos, l_max, N, d_rgb);
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
  float sum_r0 = (0.5f * 0.28209f) + 0.5f + (0.2f * 0.48860f);  // 0.238765
  float sum_g0 = (-0.2f * 0.28209f) + 0.5f + (0.2f * 0.48860f); // 0.041302
  float sum_b0 = (0.8f * 0.28209f) + 0.5f + (0.2f * 0.48860f);  // 0.323392

  // For Point 1 (x=1, y=0, z=0):
  // SH basis values = {0.28209, 0.0, 0.0, 0.48860}
  float sum_r1 = (0.1f * 0.28209f) + 0.5f + (0.4f * 0.48860f); // 0.223649
  float sum_g1 = (0.5f * 0.28209f) + 0.5f + (0.8f * 0.48860f); // 0.531925
  float sum_b1 = (0.9f * 0.28209f) + 0.5f + (0.2f * 0.48860f); // 0.351601

  const std::vector<float> expected_rgb = {sum_r0, sum_g0, sum_b0, sum_r1, sum_g1, sum_b1};

  // 7. Compare results
  for (size_t i = 0; i < h_rgb.size(); ++i) {
    ASSERT_NEAR(h_rgb[i], expected_rgb[i], 1e-4);
  }

  // 8. Cleanup
  CUDA_CHECK(cudaFree(d_xyz));
  CUDA_CHECK(cudaFree(d_sh_coefficients));
  CUDA_CHECK(cudaFree(d_rgb));
  CUDA_CHECK(cudaFree(d_band_0));
}

// Test case for the render_image function with multiple Gaussians.
// This test renders three gaussians onto a single tile and verifies the output.
TEST_F(CudaKernelTest, RenderImageMultipleGaussians) {
  // 1. Setup test parameters
  const int width = 16;
  const int height = 16;
  const int N = 3; // Number of Gaussians
  const float background_opacity = 1.0f;

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
  int *d_sorted_splats, *d_splat_range_by_tile, *d_splats_per_pixel;

  CUDA_CHECK(cudaMalloc(&d_uv, h_uv.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_opacity, h_opacity.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_conic, h_conic.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_rgb, h_rgb.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sorted_splats, h_sorted_splats.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_splat_range_by_tile, h_splat_range_by_tile.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_weight_per_pixel, h_weight_per_pixel.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_splats_per_pixel, h_weight_per_pixel.size() * sizeof(float)));
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
  render_image(d_uv, d_opacity, d_conic, d_rgb, background_opacity, d_sorted_splats, d_splat_range_by_tile, width,
               height, d_splats_per_pixel, d_weight_per_pixel, d_image);
  CUDA_CHECK(cudaDeviceSynchronize());

  // 6. Copy results back to host
  CUDA_CHECK(cudaMemcpy(h_image.data(), d_image, h_image.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // 7. Calculate expected results on the host.
  // This helper function replicates the kernel's alpha blending logic for a single pixel.
  auto calculate_expected_color = [&](float u_pixel, float v_pixel) {
    // Start with a white background and zero accumulated alpha
    float r = background_opacity, g = background_opacity, b = background_opacity;
    float alpha_accum = 0.0f;

    // Iterate through gaussians in reverse order (front to back)
    for (int i = 0; i < N; ++i) {
      const int gaussian_idx = h_sorted_splats[i];
      const float u_mean = h_uv[gaussian_idx * 2];
      const float v_mean = h_uv[gaussian_idx * 2 + 1];

      const float u_diff = u_pixel - u_mean;
      const float v_diff = v_pixel - v_mean;

      const float inv_cov00 = h_conic[gaussian_idx * 3 + 0];
      const float inv_cov01 = h_conic[gaussian_idx * 3 + 1];
      const float inv_cov11 = h_conic[gaussian_idx * 3 + 2];

      const float mh_sq =
          (inv_cov00 * u_diff * u_diff + 2.0f * inv_cov01 * u_diff * v_diff + inv_cov11 * v_diff * v_diff);

      float alpha = 0.0f;
      if (mh_sq > 0.0f) {
        float opa = (1.0f / (1.0f + expf(-h_opacity[gaussian_idx])));
        alpha = opa * expf(-0.5f * mh_sq);
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
  ASSERT_NEAR(h_image[idx_center + 0], expected_center[0], 1e-3);
  ASSERT_NEAR(h_image[idx_center + 1], expected_center[1], 1e-3);
  ASSERT_NEAR(h_image[idx_center + 2], expected_center[2], 1e-3);

  // Check a pixel far from all gaussians: (0, 0)
  // Its color should be nearly pure white background.
  int idx_corner = (0 * width + 0) * 3;
  std::vector<float> expected_corner = calculate_expected_color(0.0f, 0.0f);
  ASSERT_NEAR(h_image[idx_corner + 0], expected_corner[0], 1e-3);
  ASSERT_NEAR(h_image[idx_corner + 1], expected_corner[1], 1e-3);
  ASSERT_NEAR(h_image[idx_corner + 2], expected_corner[2], 1e-3);
  ASSERT_NEAR(h_image[idx_corner + 0], 1.0f, 1e-3); // Check against white
  ASSERT_NEAR(h_image[idx_corner + 1], 1.0f, 1e-3);
  ASSERT_NEAR(h_image[idx_corner + 2], 1.0f, 1e-3);

  // 8. Cleanup
  CUDA_CHECK(cudaFree(d_uv));
  CUDA_CHECK(cudaFree(d_opacity));
  CUDA_CHECK(cudaFree(d_conic));
  CUDA_CHECK(cudaFree(d_rgb));
  CUDA_CHECK(cudaFree(d_sorted_splats));
  CUDA_CHECK(cudaFree(d_splat_range_by_tile));
  CUDA_CHECK(cudaFree(d_weight_per_pixel));
  CUDA_CHECK(cudaFree(d_splats_per_pixel));
  CUDA_CHECK(cudaFree(d_image));
}

// Define constants for SSIM calculation.
namespace SSIMConstants {
constexpr int WINDOW_SIZE = 11;
constexpr int WINDOW_RADIUS = WINDOW_SIZE / 2;
constexpr float K1 = 0.01f;
constexpr float K2 = 0.03f;
// Dynamic range of pixel values (assuming normalized to [0,1]).
constexpr float L = 1.0f;
constexpr float C1 = (K1 * L) * (K1 * L);
constexpr float C2 = (K2 * L) * (K2 * L);
} // namespace SSIMConstants

// Test case for the fused_loss function.
// This test verifies both the combined L1/SSIM loss value and the calculated gradients.
TEST_F(CudaKernelTest, FusedLossKernel_RGB_Correctness) {
  // 1. Setup test parameters
  const int rows = 16;
  const int cols = 16;
  const int channels = 3; // RGB
  const float ssim_weight = 0.2f;
  const int num_pixels = rows * cols;
  const int total_elements = num_pixels * channels;

  // 2. Host-side data: uniform images, but different per channel
  // We use different values for R, G, and B to verify stride logic.
  // Channel 0 (R): 0.5 vs 0.6 (Slight diff)
  // Channel 1 (G): 0.4 vs 0.4 (Identical - should yield 0 L1, perfect SSIM)
  // Channel 2 (B): 0.1 vs 0.9 (Large diff)
  std::vector<float> h_pred(total_elements);
  std::vector<float> h_gt(total_elements);
  std::vector<float> h_grad(total_elements, 0.0f);

  float vals_pred[3] = {0.5f, 0.4f, 0.1f};
  float vals_gt[3] = {0.6f, 0.4f, 0.9f};

  for (int i = 0; i < num_pixels; ++i) {
    for (int c = 0; c < channels; ++c) {
      h_pred[i * channels + c] = vals_pred[c];
      h_gt[i * channels + c] = vals_gt[c];
    }
  }

  // 3. Device-side data setup
  float *d_pred, *d_gt, *d_grad;
  CUDA_CHECK(cudaMalloc(&d_pred, total_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gt, total_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_grad, total_elements * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_pred, h_pred.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gt, h_gt.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));

  // 4. Call the function
  // Note: Updated signature to include channels (3)
  float gpu_loss = fused_loss(d_pred, d_gt, rows, cols, ssim_weight, d_grad, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  // 5. Copy results back
  CUDA_CHECK(cudaMemcpy(h_grad.data(), d_grad, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

  // 6. Calculate Expected Results (Host Side)

  // SSIM Constants
  const float K1 = 0.01f;
  const float K2 = 0.03f;
  const float L_val = 1.0f;
  const float C1 = (K1 * L_val) * (K1 * L_val);
  const float C2 = (K2 * L_val) * (K2 * L_val);

  // Gaussian weights sum calculation
  const float cGauss[11] = {0.00102838f, 0.00759876f, 0.03600077f, 0.10936069f, 0.21300553f, 0.26601171f,
                            0.21300553f, 0.10936069f, 0.03600077f, 0.00759876f, 0.00102838f};
  float sum_weights_1d = 0.0f;
  for (float w : cGauss)
    sum_weights_1d += w;
  float sum_weights_2d = sum_weights_1d * sum_weights_1d;

  float expected_total_loss = 0.0f;
  float expected_grads[3] = {0.f, 0.f, 0.f};

  // Calculate per-channel expectations
  for (int c = 0; c < 3; ++c) {
    float mu_p = vals_pred[c];
    float mu_g = vals_gt[c];

    // -- Loss Calculation --
    // Uniform image -> sigma = 0, cov = 0.
    float lum_num = 2.0f * mu_p * mu_g + C1;
    float lum_den = mu_p * mu_p + mu_g * mu_g + C1;
    float expected_ssim = lum_num / lum_den; // Structure term is 1.0

    float l1_val = fabsf(mu_p - mu_g);
    float ch_loss = (1.0f - ssim_weight) * l1_val + ssim_weight * (1.0f - expected_ssim);

    expected_total_loss += ch_loss;

    // -- Gradient Calculation --

    // 1. L1 Gradient
    float l1_grad_dir = (mu_p > mu_g) ? 1.0f : -1.0f;
    // Handle the exact equality case (Channel 1) where grad is mathematically undefined/subgradient
    if (mu_p == mu_g)
      l1_grad_dir = -1.0f; // Matching the C implementation (usually > check)

    float term_l1 = (1.0f - ssim_weight) * l1_grad_dir;

    // 2. SSIM Gradient
    // d(SSIM)/d(mu_p)
    float term_num = (2.0f * mu_g) * lum_den - lum_num * (2.0f * mu_p);
    float term_den = lum_den * lum_den;
    float dSSIM_dmu1 = term_num / term_den;

    // Gradient contribution = -weight * dSSIM/dmu * sum_weights
    float term_ssim = -1.0f * dSSIM_dmu1 * sum_weights_2d;

    // 3. Normalization
    // IMPORTANT: Divided by (H * W * Channels)
    float total_grad_unscaled = term_l1 + ssim_weight * term_ssim;
    expected_grads[c] = total_grad_unscaled / (float)total_elements;
  }

  // Average the loss across channels
  expected_total_loss /= 3.0f;

  EXPECT_NEAR(gpu_loss, expected_total_loss, 1e-4) << "Loss value mismatch";

  // 7. Compare Gradients
  for (int i = 0; i < num_pixels; ++i) {
    int r = i / cols;
    int c_img = i % cols;

    // Only check center pixels to avoid boundary padding logic complexity
    if (r >= 5 && r < rows - 5 && c_img >= 5 && c_img < cols - 5) {
      for (int c = 0; c < channels; ++c) {
        float gpu_val = h_grad[i * channels + c];
        float cpu_val = expected_grads[c];

        EXPECT_NEAR(gpu_val, cpu_val, 1e-6)
            << "Gradient mismatch at pixel " << i << " channel " << c << " (Row: " << r << ", Col: " << c_img << ")";
      }
    }
  }

  // 8. Cleanup
  CUDA_CHECK(cudaFree(d_pred));
  CUDA_CHECK(cudaFree(d_gt));
  CUDA_CHECK(cudaFree(d_grad));
}

// Test case for the compute_morton_codes kernel.
TEST_F(CudaKernelTest, ComputeMortonCodes) {
  const int N = 5; // Number of points
  // Define bounding box
  const float x_min = -10.0f, x_max = 10.0f;
  const float y_min = -5.0f, y_max = 5.0f;
  const float z_min = 0.0f, z_max = 20.0f;

  // Bits per coordinate (must match kernel)
  const uint32_t BITS_PER_COORD = 21;
  const uint32_t MAX_COORD_VAL = (1 << BITS_PER_COORD) - 1;

  // Host-side input data
  // 1. Min bounds -> 0
  // 2. Max bounds -> MAX_COORD_VAL
  // 3. Center -> Middle
  // 4. Random point 1
  // 5. Random point 2
  const std::vector<float> h_xyz = {
      x_min, y_min, z_min, // Min
      x_max, y_max, z_max, // Max
      0.0f,  0.0f,  10.0f, // Center
      5.0f,  2.0f,  5.0f,  // P1
      -5.0f, -2.0f, 15.0f  // P2
  };

  std::vector<uint64_t> h_codes(N);

  // Device-side pointers
  float *d_xyz;
  uint64_t *d_codes;

  // Allocate memory
  CUDA_CHECK(cudaMalloc(&d_xyz, h_xyz.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_codes, h_codes.size() * sizeof(uint64_t)));

  // Copy inputs
  CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), h_xyz.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Run kernel
  compute_morton_codes(N, d_xyz, x_max, y_max, z_max, x_min, y_min, z_min, d_codes);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy results
  CUDA_CHECK(cudaMemcpy(h_codes.data(), d_codes, h_codes.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

  // Helper to compute expected code on host
  auto spread_bits = [&](uint64_t n) -> uint64_t {
    n &= MAX_COORD_VAL;
    n = (n | (n << 32)) & 0x1F000000FFFF;
    n = (n | (n << 16)) & 0x1F0000FF0000FF;
    n = (n | (n << 8)) & 0x100F807C0F807C0F;
    n = (n | (n << 4)) & 0x1084210842108421;
    n = (n | (n << 2)) & 0x1249249249249249;
    return n;
  };

  std::vector<uint64_t> expected_codes(N);
  for (int i = 0; i < N; ++i) {
    float x = h_xyz[i * 3 + 0];
    float y = h_xyz[i * 3 + 1];
    float z = h_xyz[i * 3 + 2];

    // Clamp to bounds (kernel doesn't explicitly clamp but logic assumes valid input,
    // however for robustness we should likely clamp if input is slightly out due to float precision,
    // but here we just pass perfect inputs).
    // Logic from kernel:
    // const uint64_t x_q = (uint64_t)((x - x_min) * (MAX_COORD_VAL / (x_max - x_min)));
    // Note: The kernel does integer cast.

    // Warning: Potential precision issues if we don't match exactly.
    // Let's rely on the fact that these are simple floats.
    // "MAX_COORD_VAL / (x_max - x_min)" might be computed in float.

    float norm_x = (x - x_min) / (x_max - x_min);
    float norm_y = (y - y_min) / (y_max - y_min);
    float norm_z = (z - z_min) / (z_max - z_min);

    // Clamp to [0, 1] to be safe
    norm_x = std::min(std::max(norm_x, 0.0f), 1.0f);
    norm_y = std::min(std::max(norm_y, 0.0f), 1.0f);
    norm_z = std::min(std::max(norm_z, 0.0f), 1.0f);

    uint64_t x_q = (uint64_t)(norm_x * MAX_COORD_VAL);
    uint64_t y_q = (uint64_t)(norm_y * MAX_COORD_VAL);
    uint64_t z_q = (uint64_t)(norm_z * MAX_COORD_VAL);

    uint64_t x_s = spread_bits(x_q);
    uint64_t y_s = spread_bits(y_q);
    uint64_t z_s = spread_bits(z_q);

    expected_codes[i] = (z_s << 2) | (y_s << 1) | x_s;
  }

  // Verify
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(h_codes[i], expected_codes[i]) << "Mismatch at index " << i << " (P=" << h_xyz[i * 3] << ","
                                             << h_xyz[i * 3 + 1] << "," << h_xyz[i * 3 + 2] << ")";
  }

  // Cleanup
  CUDA_CHECK(cudaFree(d_xyz));
  CUDA_CHECK(cudaFree(d_codes));
}
