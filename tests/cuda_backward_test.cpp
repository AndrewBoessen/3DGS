#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "gsplat_cuda/cuda_backward.cuh"

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

// Helper function to allocate memory on the CUDA device
template <typename T> T *device_alloc(size_t count) {
  T *ptr;
  cudaMalloc(&ptr, count * sizeof(T));
  return ptr;
}

// Test fixture for backward pass tests
class CudaBackwardKernelTest : public ::testing::Test {
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

// Test for project_to_screen_backward
TEST_F(CudaBackwardKernelTest, ProjectToScreenBackward) {
  const int N = 2;
  const int width = 1920;
  const int height = 1080;
  const float h = 1e-4;

  // Host data
  std::vector<float> h_xyz_c = {1.0, 2.0, 3.0, -1.0, -2.0, 4.0};
  // Proj matrix (4x4) - Identity-like for simplicity
  // P =
  // 1 0 0 0
  // 0 1 0 0
  // 0 0 0 1
  // 0 0 1 0
  // x_proj = x/z, y_proj = y/z
  // u = (x/z * 0.5 + 0.5) * width
  // v = (y/z * 0.5 + 0.5) * height
  std::vector<float> h_proj = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0};

  std::vector<float> h_uv_grad_out = {0.01, 0.02, 0.03, 0.04};
  std::vector<float> h_xyz_c_grad_in(N * 3);

  // Device data
  float *d_xyz_c = device_alloc<float>(N * 3);
  float *d_proj = device_alloc<float>(16);
  float *d_uv_grad_out = device_alloc<float>(N * 2);
  float *d_xyz_c_grad_in = device_alloc<float>(N * 3);

  CUDA_CHECK(cudaMemcpy(d_xyz_c, h_xyz_c.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_proj, h_proj.data(), 16 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_uv_grad_out, h_uv_grad_out.data(), N * 2 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_xyz_c_grad_in, 0, N * 3 * sizeof(float)));

  // Run kernel
  project_to_screen_backward(d_xyz_c, d_proj, d_uv_grad_out, N, width, height, d_xyz_c_grad_in);

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_xyz_c_grad_in.data(), d_xyz_c_grad_in, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  // Numerical gradient check
  auto forward_proj = [&](const std::vector<float> &xyz_c, const std::vector<float> &proj) {
    std::vector<float> uv(N * 2);
    for (int i = 0; i < N; ++i) {
      float x = xyz_c[i * 3 + 0];
      float y = xyz_c[i * 3 + 1];
      float z = xyz_c[i * 3 + 2];
      // With our custom Proj:
      uv[i * 2 + 0] = (x / z * 0.5f + 0.5f) * width;
      uv[i * 2 + 1] = (y / z * 0.5f + 0.5f) * height;
    }
    return uv;
  };

  // Check grad w.r.t xyz_c
  for (int i = 0; i < N * 3; ++i) {
    std::vector<float> xyz_c_p = h_xyz_c;
    xyz_c_p[i] += h;
    std::vector<float> xyz_c_m = h_xyz_c;
    xyz_c_m[i] -= h;
    auto uv_p = forward_proj(xyz_c_p, h_proj);
    auto uv_m = forward_proj(xyz_c_m, h_proj);
    float numerical_grad = 0;
    for (int j = 0; j < N * 2; ++j)
      numerical_grad += (uv_p[j] - uv_m[j]) / (2 * h) * h_uv_grad_out[j];
    EXPECT_NEAR(h_xyz_c_grad_in[i], numerical_grad, 1e-1);
  }

  CUDA_CHECK(cudaFree(d_xyz_c));
  CUDA_CHECK(cudaFree(d_proj));
  CUDA_CHECK(cudaFree(d_uv_grad_out));
  CUDA_CHECK(cudaFree(d_xyz_c_grad_in));
}

// Test for compute_camera_space_points_backward
TEST_F(CudaBackwardKernelTest, ComputeCameraSpacePointsBackward) {
  const int N = 1;
  const float h = 1e-4;

  // Host data
  std::vector<float> h_xyz_w = {1.0, 2.0, 3.0};
  // View matrix (4x4)
  std::vector<float> h_view = {0.8, -0.6, 0.0, 0.1, 0.6, 0.8, 0.0, 0.2, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0, 1.0};
  std::vector<float> h_xyz_c_grad_in = {0.1, 0.2, 0.3};
  std::vector<float> h_xyz_w_grad_in(N * 3);

  // Device data
  auto d_xyz_w = device_alloc<float>(N * 3);
  auto d_view = device_alloc<float>(16);
  auto d_xyz_c_grad_in = device_alloc<float>(N * 3);
  auto d_xyz_w_grad_in = device_alloc<float>(N * 3);

  CUDA_CHECK(cudaMemcpy(d_xyz_w, h_xyz_w.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_view, h_view.data(), 16 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_xyz_c_grad_in, h_xyz_c_grad_in.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));

  compute_camera_space_points_backward(d_xyz_w, d_view, d_xyz_c_grad_in, N, d_xyz_w_grad_in);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_xyz_w_grad_in.data(), d_xyz_w_grad_in, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  auto forward_ext = [&](const std::vector<float> &xyz_w, const std::vector<float> &view) {
    std::vector<float> xyz_c(N * 3);
    for (int i = 0; i < N; ++i) {
      xyz_c[i * 3 + 0] = view[0] * xyz_w[i * 3 + 0] + view[1] * xyz_w[i * 3 + 1] + view[2] * xyz_w[i * 3 + 2] + view[3];
      xyz_c[i * 3 + 1] = view[4] * xyz_w[i * 3 + 0] + view[5] * xyz_w[i * 3 + 1] + view[6] * xyz_w[i * 3 + 2] + view[7];
      xyz_c[i * 3 + 2] =
          view[8] * xyz_w[i * 3 + 0] + view[9] * xyz_w[i * 3 + 1] + view[10] * xyz_w[i * 3 + 2] + view[11];
    }
    return xyz_c;
  };

  for (int i = 0; i < N * 3; ++i) {
    std::vector<float> xyz_w_p = h_xyz_w;
    xyz_w_p[i] += h;
    std::vector<float> xyz_w_m = h_xyz_w;
    xyz_w_m[i] -= h;
    auto xyz_c_p = forward_ext(xyz_w_p, h_view);
    auto xyz_c_m = forward_ext(xyz_w_m, h_view);
    float numerical_grad = 0;
    for (int j = 0; j < N * 3; ++j)
      numerical_grad += (xyz_c_p[j] - xyz_c_m[j]) / (2 * h) * h_xyz_c_grad_in[j];
    EXPECT_NEAR(h_xyz_w_grad_in[i], numerical_grad, 1e-1);
  }

  CUDA_CHECK(cudaFree(d_xyz_w));
  CUDA_CHECK(cudaFree(d_view));
  CUDA_CHECK(cudaFree(d_xyz_c_grad_in));
  CUDA_CHECK(cudaFree(d_xyz_w_grad_in));
}

// Test for compute_projection_jacobian_backward
TEST_F(CudaBackwardKernelTest, ProjectionJacobianBackward) {
  const int N = 2;
  const float h = 1e-4;

  // Host data
  std::vector<float> h_xyz_c = {1.0, 2.0, 3.0, -1.0, -2.0, 4.0};
  // Proj matrix (4x4) - Identity-like for simplicity
  // P =
  // 1 0 0 0
  // 0 1 0 0
  // 0 0 0 1
  // 0 0 1 0
  std::vector<float> h_proj = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0};

  std::vector<float> h_J_grad_in = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
  std::vector<float> h_xyz_c_grad_out(N * 3);

  // Device data
  float *d_xyz_c = device_alloc<float>(N * 3);
  float *d_J_grad_in = device_alloc<float>(N * 6);
  float *d_xyz_c_grad_out = device_alloc<float>(N * 3);

  // Focal length and tan_fov derived from Identity-like proj where P00=1, P11=1, P32=1
  const float focal_x = 1.0f;
  const float focal_y = 1.0f;
  const float tan_fovx = 1.0f;
  const float tan_fovy = 1.0f;

  CUDA_CHECK(cudaMemcpy(d_xyz_c, h_xyz_c.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_J_grad_in, h_J_grad_in.data(), N * 6 * sizeof(float), cudaMemcpyHostToDevice));

  // Run kernel
  compute_projection_jacobian_backward(d_xyz_c, focal_x, focal_y, tan_fovx, tan_fovy, d_J_grad_in, N, d_xyz_c_grad_out);

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_xyz_c_grad_out.data(), d_xyz_c_grad_out, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  // Numerical gradient check
  auto forward_jacobian = [&](const std::vector<float> &xyz_c) {
    std::vector<float> J(N * 6);
    for (int i = 0; i < N; ++i) {
      float x = xyz_c[i * 3 + 0];
      float y = xyz_c[i * 3 + 1];
      float z = xyz_c[i * 3 + 2];
      float z_inv = 1.0f / z;
      float z_inv2 = z_inv * z_inv;

      // Jacobian: du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz
      // With our simple Proj:
      // J = [ 1/z, 0, -x/z^2 ]
      //     [ 0, 1/z, -y/z^2 ]
      J[i * 6 + 0] = z_inv;       // du/dx
      J[i * 6 + 1] = 0.0f;        // du/dy
      J[i * 6 + 2] = -x * z_inv2; // du/dz
      J[i * 6 + 3] = 0.0f;        // dv/dx
      J[i * 6 + 4] = z_inv;       // dv/dy
      J[i * 6 + 5] = -y * z_inv2; // dv/dz
    }
    return J;
  };

  // Check grad w.r.t xyz_c
  for (int i = 0; i < N * 3; ++i) {
    std::vector<float> xyz_c_p = h_xyz_c;
    xyz_c_p[i] += h;
    std::vector<float> xyz_c_m = h_xyz_c;
    xyz_c_m[i] -= h;
    auto J_p = forward_jacobian(xyz_c_p);
    auto J_m = forward_jacobian(xyz_c_m);
    float numerical_grad = 0;
    for (int j = 0; j < N * 6; ++j)
      numerical_grad += (J_p[j] - J_m[j]) / (2 * h) * h_J_grad_in[j];
    EXPECT_NEAR(h_xyz_c_grad_out[i], numerical_grad, 1e-1);
  }

  CUDA_CHECK(cudaFree(d_xyz_c));
  CUDA_CHECK(cudaFree(d_J_grad_in));
  CUDA_CHECK(cudaFree(d_xyz_c_grad_out));
}

// Test for compute_conic_backward
TEST_F(CudaBackwardKernelTest, ConicBackward) {
  const int N = 1;
  const float h = 1e-4f;

  // Host data
  std::vector<float> h_J = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  std::vector<float> h_sigma_world = {1.0f, 0.1f, 0.2f, 2.0f, 0.3f, 3.0f}; // xx, xy, xz, yy, yz, zz
  // View matrix (4x4)
  std::vector<float> h_view = {0.8f, -0.6f, 0.0f, 0.1f, 0.6f, 0.8f, 0.0f, 0.2f,
                               0.0f, 0.0f,  1.0f, 0.3f, 0.0f, 0.0f, 0.0f, 1.0f};
  std::vector<float> h_conic_grad_out = {0.5f, -0.2f, 0.8f};
  std::vector<float> h_J_grad_in(N * 6);
  std::vector<float> h_sigma_world_grad_in(N * 6); // Kernel has i*6 indexing, so allocate 6 floats

  // Compute h_conic (inverse covariance) for the test
  auto compute_conic_val = [&](const std::vector<float> &J_in, const std::vector<float> &sigma_in,
                               const std::vector<float> &view_in) {
    const float *J = J_in.data();
    // Reconstruct full 3x3 sigma from 6 params
    float S[9];
    S[0] = sigma_in[0]; // xx
    S[1] = sigma_in[1]; // xy
    S[2] = sigma_in[2]; // xz
    S[3] = sigma_in[1]; // yx
    S[4] = sigma_in[3]; // yy
    S[5] = sigma_in[4]; // yz
    S[6] = sigma_in[2]; // zx
    S[7] = sigma_in[4]; // zy
    S[8] = sigma_in[5]; // zz
    const float W[9] = {view_in[0], view_in[1], view_in[2], view_in[4], view_in[5],
                        view_in[6], view_in[8], view_in[9], view_in[10]};

    // JW = J @ W (2x3)
    float JW[6];
    JW[0] = J[0] * W[0] + J[1] * W[3] + J[2] * W[6];
    JW[1] = J[0] * W[1] + J[1] * W[4] + J[2] * W[7];
    JW[2] = J[0] * W[2] + J[1] * W[5] + J[2] * W[8];
    JW[3] = J[3] * W[0] + J[4] * W[3] + J[5] * W[6];
    JW[4] = J[3] * W[1] + J[4] * W[4] + J[5] * W[7];
    JW[5] = J[3] * W[2] + J[4] * W[5] + J[5] * W[8];

    // temp = JW @ S (2x3)
    float temp[6];
    temp[0] = JW[0] * S[0] + JW[1] * S[3] + JW[2] * S[6];
    temp[1] = JW[0] * S[1] + JW[1] * S[4] + JW[2] * S[7];
    temp[2] = JW[0] * S[2] + JW[1] * S[5] + JW[2] * S[8];
    temp[3] = JW[3] * S[0] + JW[4] * S[3] + JW[5] * S[6];
    temp[4] = JW[3] * S[1] + JW[4] * S[4] + JW[5] * S[7];
    temp[5] = JW[3] * S[2] + JW[4] * S[5] + JW[5] * S[8];

    // cov = temp @ JW.T (2x2 symmetric)
    float cov00 = temp[0] * JW[0] + temp[1] * JW[1] + temp[2] * JW[2] + 0.3f;
    float cov01 = temp[0] * JW[3] + temp[1] * JW[4] + temp[2] * JW[5];
    float cov11 = temp[3] * JW[3] + temp[4] * JW[4] + temp[5] * JW[5] + 0.3f;

    // Invert
    float det = cov00 * cov11 - cov01 * cov01;
    float inv_det = 1.0f / det;
    std::vector<float> conic(3);
    conic[0] = cov11 * inv_det;
    conic[1] = -cov01 * inv_det;
    conic[2] = cov00 * inv_det;
    return conic;
  };

  std::vector<float> h_conic = compute_conic_val(h_J, h_sigma_world, h_view);

  // Device data
  auto d_J = device_alloc<float>(N * 6);
  auto d_sigma_world = device_alloc<float>(N * 6);
  auto d_view = device_alloc<float>(16);
  auto d_conic = device_alloc<float>(N * 3);
  auto d_conic_grad_out = device_alloc<float>(N * 3);
  auto d_J_grad_in = device_alloc<float>(N * 6);
  auto d_sigma_world_grad_in = device_alloc<float>(N * 6);

  CUDA_CHECK(cudaMemcpy(d_J, h_J.data(), N * 6 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sigma_world, h_sigma_world.data(), N * 6 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_view, h_view.data(), 16 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_conic, h_conic.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_conic_grad_out, h_conic_grad_out.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_J_grad_in, 0, N * 6 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_sigma_world_grad_in, 0, N * 6 * sizeof(float)));

  // Run kernel
  compute_conic_backward(d_J, d_sigma_world, d_view, d_conic, d_conic_grad_out, N, d_J_grad_in, d_sigma_world_grad_in);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_J_grad_in.data(), d_J_grad_in, N * 6 * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h_sigma_world_grad_in.data(), d_sigma_world_grad_in, N * 6 * sizeof(float), cudaMemcpyDeviceToHost));

  auto compute_loss = [&](const std::vector<float> &conic) {
    return conic[0] * h_conic_grad_out[0] + 2.0f * conic[1] * h_conic_grad_out[1] + conic[2] * h_conic_grad_out[2];
  };

  // Check grad w.r.t. J
  for (int i = 0; i < N * 6; ++i) {
    std::vector<float> J_p = h_J;
    J_p[i] += h;
    std::vector<float> J_m = h_J;
    J_m[i] -= h;
    auto loss_p = compute_loss(compute_conic_val(J_p, h_sigma_world, h_view));
    auto loss_m = compute_loss(compute_conic_val(J_m, h_sigma_world, h_view));
    float numerical_grad = (loss_p - loss_m) / (2.0f * h);
    EXPECT_NEAR(h_J_grad_in[i], numerical_grad, 1e-2);
  }

  // Reconstruct full symmetric gradient for sigma from kernel output (which is 6 params)
  // The kernel accumulates gradients into the 6 unique elements.
  // dL/dS_ij_full = dL/dS_ij_stored (if i==j)
  // dL/dS_ij_full = 0.5 * dL/dS_ij_stored (if i!=j, because stored accumulates both ij and ji)
  std::vector<float> h_sigma_grad_analytic_full(6);
  h_sigma_grad_analytic_full[0] = h_sigma_world_grad_in[0]; // xx
  h_sigma_grad_analytic_full[1] = h_sigma_world_grad_in[1]; // xy
  h_sigma_grad_analytic_full[2] = h_sigma_world_grad_in[2]; // xz
  h_sigma_grad_analytic_full[3] = h_sigma_world_grad_in[3]; // yy
  h_sigma_grad_analytic_full[4] = h_sigma_world_grad_in[4]; // yz
  h_sigma_grad_analytic_full[5] = h_sigma_world_grad_in[5]; // zz

  // Check grad w.r.t. sigma_world (6 params)
  for (int i = 0; i < N * 6; ++i) {
    std::vector<float> sigma_p = h_sigma_world;
    sigma_p[i] += h;
    std::vector<float> sigma_m = h_sigma_world;
    sigma_m[i] -= h;
    auto loss_p = compute_loss(compute_conic_val(h_J, sigma_p, h_view));
    auto loss_m = compute_loss(compute_conic_val(h_J, sigma_m, h_view));
    float numerical_grad = (loss_p - loss_m) / (2.0f * h);
    EXPECT_NEAR(h_sigma_grad_analytic_full[i], numerical_grad, 1e-2);
  }

  CUDA_CHECK(cudaFree(d_J));
  CUDA_CHECK(cudaFree(d_sigma_world));
  CUDA_CHECK(cudaFree(d_view));
  CUDA_CHECK(cudaFree(d_conic));
  CUDA_CHECK(cudaFree(d_conic_grad_out));
  CUDA_CHECK(cudaFree(d_J_grad_in));
  CUDA_CHECK(cudaFree(d_sigma_world_grad_in));
}

TEST_F(CudaBackwardKernelTest, SigmaBackward) {
  const int N = 1;
  const float h = 1e-4;

  // Host data
  std::vector<float> h_q = {0.70710678, 0.70710678, 0.0, 0.0}; // Gaussian 1: 90 deg rot around X
  std::vector<float> h_s = {-0.1, -0.2, -0.3};
  std::vector<float> h_dSigma_in = {-0.1, -0.2, -0.3, -0.4, -0.5, -0.6}; // xx, xy, xz, yy, yz, zz
  std::vector<float> h_dQ_in(N * 4);
  std::vector<float> h_dS_in(N * 3);

  // Device data
  auto d_q = device_alloc<float>(N * 4);
  auto d_s = device_alloc<float>(N * 3);
  auto d_dSigma_in = device_alloc<float>(N * 6);
  auto d_dQ_in = device_alloc<float>(N * 4);
  auto d_dS_in = device_alloc<float>(N * 3);

  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), N * 4 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_s, h_s.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dSigma_in, h_dSigma_in.data(), N * 6 * sizeof(float), cudaMemcpyHostToDevice));

  // Run kernel
  compute_sigma_backward(d_q, d_s, d_dSigma_in, N, d_dQ_in, d_dS_in);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_dQ_in.data(), d_dQ_in, N * 4 * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_dS_in.data(), d_dS_in, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  // Numerical gradient check
  auto forward_sigma = [&](const std::vector<float> &q_in, const std::vector<float> &s_in) {
    std::vector<float> sigma(N * 6);
    for (int i = 0; i < N; ++i) {
      float qw = q_in[i * 4 + 0];
      float qx = q_in[i * 4 + 1];
      float qy = q_in[i * 4 + 2];
      float qz = q_in[i * 4 + 3];

      float norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz) + 1e-8f;
      float w = qw / norm;
      float x = qx / norm;
      float y = qy / norm;
      float z = qz / norm;

      float S_x = std::exp(s_in[i * 3 + 0]);
      float S_y = std::exp(s_in[i * 3 + 1]);
      float S_z = std::exp(s_in[i * 3 + 2]);

      float R[9];
      R[0] = 1.0f - 2.0f * (y * y + z * z);
      R[1] = 2.0f * (x * y - w * z);
      R[2] = 2.0f * (x * z + w * y);
      R[3] = 2.0f * (x * y + w * z);
      R[4] = 1.0f - 2.0f * (x * x + z * z);
      R[5] = 2.0f * (y * z - w * x);
      R[6] = 2.0f * (x * z - w * y);
      R[7] = 2.0f * (y * z + w * x);
      R[8] = 1.0f - 2.0f * (x * x + y * y);

      float M[9];
      M[0] = R[0] * S_x;
      M[1] = R[1] * S_y;
      M[2] = R[2] * S_z;
      M[3] = R[3] * S_x;
      M[4] = R[4] * S_y;
      M[5] = R[5] * S_z;
      M[6] = R[6] * S_x;
      M[7] = R[7] * S_y;
      M[8] = R[8] * S_z;

      // Sigma = M * M^T
      // Store 6 params: xx, xy, xz, yy, yz, zz
      sigma[i * 6 + 0] = M[0] * M[0] + M[1] * M[1] + M[2] * M[2];
      sigma[i * 6 + 1] = M[0] * M[3] + M[1] * M[4] + M[2] * M[5];
      sigma[i * 6 + 2] = M[0] * M[6] + M[1] * M[7] + M[2] * M[8];
      sigma[i * 6 + 3] = M[3] * M[3] + M[4] * M[4] + M[5] * M[5];
      sigma[i * 6 + 4] = M[3] * M[6] + M[4] * M[7] + M[5] * M[8];
      sigma[i * 6 + 5] = M[6] * M[6] + M[7] * M[7] + M[8] * M[8];
    }
    return sigma;
  };

  auto compute_loss = [&](const std::vector<float> &sigma) {
    float loss = 0.0f;
    for (size_t i = 0; i < sigma.size(); ++i) {
      loss += sigma[i] * h_dSigma_in[i];
    }
    return loss;
  };

  // Check grad w.r.t q
  for (int i = 0; i < N * 4; ++i) {
    std::vector<float> q_p = h_q;
    q_p[i] += h;
    std::vector<float> q_m = h_q;
    q_m[i] -= h;

    auto sigma_p = forward_sigma(q_p, h_s);
    auto sigma_m = forward_sigma(q_m, h_s);

    float loss_p = compute_loss(sigma_p);
    float loss_m = compute_loss(sigma_m);

    float numerical_grad = (loss_p - loss_m) / (2 * h);
    EXPECT_NEAR(h_dQ_in[i], numerical_grad, 1e-2);
  }

  // Check grad w.r.t s
  for (int i = 0; i < N * 3; ++i) {
    std::vector<float> s_p = h_s;
    s_p[i] += h;
    std::vector<float> s_m = h_s;
    s_m[i] -= h;

    auto sigma_p = forward_sigma(h_q, s_p);
    auto sigma_m = forward_sigma(h_q, s_m);

    float loss_p = compute_loss(sigma_p);
    float loss_m = compute_loss(sigma_m);

    float numerical_grad = (loss_p - loss_m) / (2 * h);
    EXPECT_NEAR(h_dS_in[i], numerical_grad, 1e-2);
  }

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_s));
  CUDA_CHECK(cudaFree(d_dSigma_in));
  CUDA_CHECK(cudaFree(d_dQ_in));
  CUDA_CHECK(cudaFree(d_dS_in));
}

// Test for precompute_spherical_harmonics_backward
TEST_F(CudaBackwardKernelTest, SphericalHarmonicsBackward) {
  const int N = 1;
  const int l_max = 2;
  const int n_coeffs = (l_max + 1) * (l_max + 1);
  const float h = 1e-4f;

  // Host data
  std::vector<float> h_xyz_c = {1.0f, 1.0f, 0.5f}; // Roughly normalized vector
  std::vector<float> h_rgb_grad_out = {0.1f, -0.2f, 0.3f};

  std::vector<float> h_rgb_vals(N * 3);
  std::vector<float> h_sh_rest(N * (n_coeffs - 1) * 3);

  // Fill them similarly to before for consistency in checking
  for (int i = 0; i < N * 3; ++i) {
    h_rgb_vals[i] = 0.5f; // Band 0 values
  }
  for (int i = 0; i < h_sh_rest.size(); ++i) {
    h_sh_rest[i] = (i % 10) * 0.05f - 0.2f;
  }

  std::vector<float> h_sh_grad_in(N * n_coeffs * 3);
  std::vector<float> h_xyz_c_grad_in(N * 3);

  // Device data
  auto d_xyz_c = device_alloc<float>(N * 3);
  auto d_rgb_grad_out = device_alloc<float>(N * 3);
  auto d_rgb_vals = device_alloc<float>(N * 3);
  auto d_sh_rest = device_alloc<float>(N * (n_coeffs - 1) * 3);

  auto d_sh_grad_in = device_alloc<float>(N * (n_coeffs - 1) * 3);
  auto d_band_0_grad = device_alloc<float>(N * 3);
  auto d_xyz_c_grad_in = device_alloc<float>(N * 3);

  // Allocate dummy campos at origin
  float3 campos = {0.0f, 0.0f, 0.0f};

  CUDA_CHECK(cudaMemcpy(d_xyz_c, h_xyz_c.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rgb_grad_out, h_rgb_grad_out.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rgb_vals, h_rgb_vals.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sh_rest, h_sh_rest.data(), N * (n_coeffs - 1) * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_xyz_c_grad_in, 0, N * 3 * sizeof(float)));

  // Run kernel
  precompute_spherical_harmonics_backward(d_xyz_c, d_rgb_vals, d_sh_rest, campos, d_rgb_grad_out, l_max, N,
                                          d_sh_grad_in, d_band_0_grad, d_xyz_c_grad_in);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_sh_grad_in.data() + N * 3, d_sh_grad_in, N * (n_coeffs - 1) * 3 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_sh_grad_in.data(), d_band_0_grad, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_xyz_c_grad_in.data(), d_xyz_c_grad_in, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  // Numerical gradient check
  auto forward_sh_rgb = [&](const std::vector<float> &rgb_vals, const std::vector<float> &sh_rest,
                            const std::vector<float> &xyz_c) {
    std::vector<float> logits(N * 3, 0.0f);
    std::vector<float> sh_vals(n_coeffs);

    for (int i = 0; i < N; ++i) {
      float x_ = xyz_c[i * 3 + 0];
      float y_ = xyz_c[i * 3 + 1];
      float z_ = xyz_c[i * 3 + 2];
      float norm = std::sqrt(x_ * x_ + y_ * y_ + z_ * z_);
      float x = x_ / norm, y = y_ / norm, z = z_ / norm;

      // Real Spherical Harmonics basis functions
      const float C0 = 0.28209479177387814f;
      const float C1 = 0.4886025119029199f;
      const float C2 = 1.0925484305920792f;
      const float C3 = 0.31539156525252005f;
      const float C4 = 0.5462742152960399f;

      sh_vals[0] = C0;
      sh_vals[1] = C1 * y;
      sh_vals[2] = C1 * z;
      sh_vals[3] = C1 * x;
      sh_vals[4] = C2 * x * y;
      sh_vals[5] = C2 * y * z;
      sh_vals[6] = C3 * (3.0f * z * z - 1.0f);
      sh_vals[7] = C2 * x * z;
      sh_vals[8] = C4 * (x * x - y * y);

      // Band 0
      logits[i * 3 + 0] += rgb_vals[i * 3 + 0] * sh_vals[0];
      logits[i * 3 + 1] += rgb_vals[i * 3 + 1] * sh_vals[0];
      logits[i * 3 + 2] += rgb_vals[i * 3 + 2] * sh_vals[0];

      // Higher Bands
      const float *point_sh_rest = &sh_rest[i * (n_coeffs - 1) * 3];
      for (int j = 1; j < n_coeffs; ++j) {
        int idx_in_rest = (j - 1);
        logits[i * 3 + 0] += point_sh_rest[idx_in_rest * 3 + 0] * sh_vals[j];
        logits[i * 3 + 1] += point_sh_rest[idx_in_rest * 3 + 1] * sh_vals[j];
        logits[i * 3 + 2] += point_sh_rest[idx_in_rest * 3 + 2] * sh_vals[j];
      }
    }
    return logits;
  };

  auto compute_loss = [&](const std::vector<float> &logits) {
    double loss = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
      loss += static_cast<double>(logits[i]) * h_rgb_grad_out[i];
    }
    return loss;
  };

  // Check grad w.r.t xyz_c
  for (int i = 0; i < N * 3; ++i) {
    std::vector<float> xyz_c_p = h_xyz_c;
    xyz_c_p[i] += h;
    std::vector<float> xyz_c_m = h_xyz_c;
    xyz_c_m[i] -= h;

    auto logits_p = forward_sh_rgb(h_rgb_vals, h_sh_rest, xyz_c_p);
    auto logits_m = forward_sh_rgb(h_rgb_vals, h_sh_rest, xyz_c_m);

    double loss_p = compute_loss(logits_p);
    double loss_m = compute_loss(logits_m);

    float numerical_grad = (loss_p - loss_m) / (2.0f * h);
    EXPECT_NEAR(h_xyz_c_grad_in[i], numerical_grad, 1e-2);
  }

  CUDA_CHECK(cudaFree(d_xyz_c));
  CUDA_CHECK(cudaFree(d_rgb_grad_out));
  CUDA_CHECK(cudaFree(d_rgb_vals));
  CUDA_CHECK(cudaFree(d_sh_rest));
  CUDA_CHECK(cudaFree(d_sh_grad_in));
  CUDA_CHECK(cudaFree(d_band_0_grad));
  CUDA_CHECK(cudaFree(d_xyz_c_grad_in));
}

// Test for render_image_backward
TEST_F(CudaBackwardKernelTest, RenderBackward) {
  const int image_width = 16, image_height = 16;
  const int N = 3; // Number of Gaussians
  const float h = 1e-6f;

  // Host data
  std::vector<float> h_uvs = {4.5f, 4.5f, 8.5f, 8.5f, 12.5f, 12.5f};
  std::vector<float> h_opacity = {10.0f, 10.0f, 10.0f};
  std::vector<float> h_conic = {2.0f, 0.1f, 2.0f, 2.0f, 0.1f, 2.0f, 2.0f, 0.1f, 2.0f}; // Gaussian 1
  std::vector<float> h_rgb = {0.5f, 0.2f, 0.2f, 0.2f, 0.2f, 0.5f, 0.2f, 0.5f, 0.2f};   // Gaussian 1
  const float background_opacity = 0.5f;
  std::vector<float> h_grad_image(image_width * image_height * 3);
  for (size_t i = 0; i < h_grad_image.size(); ++i)
    h_grad_image[i] = 1e-3f;

  // Data that is computed during the forward pass
  std::vector<int> h_sorted_splats = {0, 1, 2};
  std::vector<int> h_splat_range_by_tile = {0, 3}; // One tile [0, 3)
  std::vector<int> h_num_splats_per_pixel(image_width * image_height);
  std::vector<float> h_final_weight_per_pixel(image_width * image_height);

  // CPU forward pass to generate required inputs for backward pass
  auto forward_render = [&](const std::vector<float> &uvs, const std::vector<float> &opacity,
                            const std::vector<float> &conic, const std::vector<float> &rgb,
                            std::vector<int> &num_splats_per_pixel, std::vector<float> &final_weight_per_pixel) {
    std::vector<float> image(image_width * image_height * 3);
    for (int v_splat = 0; v_splat < image_height; ++v_splat) {
      for (int u_splat = 0; u_splat < image_width; ++u_splat) {
        int splat_count = 0;
        float T = 1.0f;
        float pixel_rgb[3] = {0.0f, 0.0f, 0.0f};

        // Get splat range for this tile.
        const int splat_idx_start = h_splat_range_by_tile[0];
        const int splat_idx_end = h_splat_range_by_tile[1];

        bool done = false;

        for (int splat_idx = splat_idx_start; splat_idx < splat_idx_end; ++splat_idx) {
          const int i = h_sorted_splats[splat_idx];

          const float u_mean = uvs[i * 2 + 0];
          const float v_mean = uvs[i * 2 + 1];
          const float u_diff = (float)u_splat - u_mean;
          const float v_diff = (float)v_splat - v_mean;

          const float opa = 1.0f / (1.0f + expf(-opacity[i]));

          const float inv_cov00 = conic[i * 3 + 0];
          const float inv_cov01 = conic[i * 3 + 1];
          const float inv_cov11 = conic[i * 3 + 2];

          const float power = fminf(0.0f, -0.5f * (inv_cov00 * u_diff * u_diff + 2.0f * inv_cov01 * u_diff * v_diff +
                                                   inv_cov11 * v_diff * v_diff));

          float norm_prob = std::exp(power);

          // Match kernel: opacity logit -> sigmoid
          float alpha = std::min(0.99f, opa * norm_prob);
          alpha = (alpha > 0.00392156862f) ? !done * alpha : 0.0f;

          const float test_T = T * (1.0f - alpha);
          done = test_T < 0.0001f;

          const float weight = alpha * T;

          pixel_rgb[0] += rgb[i * 3 + 0] * weight;
          pixel_rgb[1] += rgb[i * 3 + 1] * weight;
          pixel_rgb[2] += rgb[i * 3 + 2] * weight;

          T = test_T;
          splat_count += !done;
        }

        int pixel_idx = v_splat * image_width + u_splat;
        image[pixel_idx * 3 + 0] = pixel_rgb[0] + T * background_opacity;
        image[pixel_idx * 3 + 1] = pixel_rgb[1] + T * background_opacity;
        image[pixel_idx * 3 + 2] = pixel_rgb[2] + T * background_opacity;

        num_splats_per_pixel[pixel_idx] = splat_count;
        final_weight_per_pixel[pixel_idx] = T;
      }
    }
    return image;
  };

  forward_render(h_uvs, h_opacity, h_conic, h_rgb, h_num_splats_per_pixel, h_final_weight_per_pixel);

  // Device data
  auto d_uvs = device_alloc<float>(N * 2);
  auto d_opacity = device_alloc<float>(N);
  auto d_conic = device_alloc<float>(N * 3);
  auto d_rgb = device_alloc<float>(N * 3);
  auto d_sorted_splats = device_alloc<int>(N);
  auto d_splat_range_by_tile = device_alloc<int>(2);
  auto d_num_splats_per_pixel = device_alloc<int>(image_width * image_height);
  auto d_final_weight_per_pixel = device_alloc<float>(image_width * image_height);
  auto d_grad_image = device_alloc<float>(image_width * image_height * 3);

  auto d_grad_uv = device_alloc<float>(N * 2);
  auto d_grad_opacity = device_alloc<float>(N);
  auto d_grad_conic = device_alloc<float>(N * 3);
  auto d_grad_rgb = device_alloc<float>(N * 3);

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_uvs, h_uvs.data(), h_uvs.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_opacity, h_opacity.data(), h_opacity.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_conic, h_conic.data(), h_conic.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb.data(), h_rgb.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sorted_splats, h_sorted_splats.data(), h_sorted_splats.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_splat_range_by_tile, h_splat_range_by_tile.data(), h_splat_range_by_tile.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_num_splats_per_pixel, h_num_splats_per_pixel.data(),
                        h_num_splats_per_pixel.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_final_weight_per_pixel, h_final_weight_per_pixel.data(),
                        h_final_weight_per_pixel.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_grad_image, h_grad_image.data(), h_grad_image.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Zero out gradient buffers
  CUDA_CHECK(cudaMemset(d_grad_uv, 0, N * 2 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_grad_opacity, 0, N * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_grad_conic, 0, N * 3 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_grad_rgb, 0, N * 3 * sizeof(float)));

  // Run kernel
  render_image_backward(d_uvs, d_opacity, d_conic, d_rgb, background_opacity, d_sorted_splats, d_splat_range_by_tile,
                        d_num_splats_per_pixel, d_final_weight_per_pixel, d_grad_image, image_width, image_height,
                        d_grad_rgb, d_grad_opacity, d_grad_uv, d_grad_conic, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy results back
  std::vector<float> h_grad_uv(N * 2), h_grad_opacity(N), h_grad_conic(N * 3), h_grad_rgb(N * 3);
  CUDA_CHECK(cudaMemcpy(h_grad_uv.data(), d_grad_uv, h_grad_uv.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h_grad_opacity.data(), d_grad_opacity, h_grad_opacity.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h_grad_conic.data(), d_grad_conic, h_grad_conic.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_grad_rgb.data(), d_grad_rgb, h_grad_rgb.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // Numerical gradient check
  auto compute_loss = [&](const std::vector<float> &image) {
    double loss = 0.0;
    for (size_t i = 0; i < image.size(); ++i) {
      loss += (double)image[i] * h_grad_image[i];
    }
    return loss;
  };

  // Gradients for uvs
  for (int i = 0; i < N * 2; ++i) {
    std::vector<float> uvs_p = h_uvs, uvs_m = h_uvs;
    uvs_p[i] += h;
    uvs_m[i] -= h;
    auto image_p = forward_render(uvs_p, h_opacity, h_conic, h_rgb, h_num_splats_per_pixel, h_final_weight_per_pixel);
    auto image_m = forward_render(uvs_m, h_opacity, h_conic, h_rgb, h_num_splats_per_pixel, h_final_weight_per_pixel);
    double loss_p = compute_loss(image_p);
    double loss_m = compute_loss(image_m);
    float num_grad = (loss_p - loss_m) / (2.0f * h);
    float norm_factor = (i % 2 == 0) ? (0.5f * image_width) : (0.5f * image_height);
    num_grad *= norm_factor;
    EXPECT_NEAR(h_grad_uv[i], num_grad, 1e-3);
  }

  // Gradients for opacity
  for (int i = 0; i < N; ++i) {
    std::vector<float> opacity_p = h_opacity, opacity_m = h_opacity;
    opacity_p[i] += h;
    opacity_m[i] -= h;
    auto image_p = forward_render(h_uvs, opacity_p, h_conic, h_rgb, h_num_splats_per_pixel, h_final_weight_per_pixel);
    auto image_m = forward_render(h_uvs, opacity_m, h_conic, h_rgb, h_num_splats_per_pixel, h_final_weight_per_pixel);
    double loss_p = compute_loss(image_p);
    double loss_m = compute_loss(image_m);
    float num_grad = (loss_p - loss_m) / (2.0f * h);
    EXPECT_NEAR(h_grad_opacity[i], num_grad, 1e-3);
  }

  // Gradients for conic
  for (int i = 0; i < N * 3; ++i) {
    std::vector<float> conic_p = h_conic, conic_m = h_conic;
    conic_p[i] += h;
    conic_m[i] -= h;
    auto image_p = forward_render(h_uvs, h_opacity, conic_p, h_rgb, h_num_splats_per_pixel, h_final_weight_per_pixel);
    auto image_m = forward_render(h_uvs, h_opacity, conic_m, h_rgb, h_num_splats_per_pixel, h_final_weight_per_pixel);
    double loss_p = compute_loss(image_p);
    double loss_m = compute_loss(image_m);
    float num_grad = (loss_p - loss_m) / (2.0f * h);
    EXPECT_NEAR(h_grad_conic[i], num_grad, 1e-3);
  }

  // Gradients for rgb
  for (int i = 0; i < N * 3; ++i) {
    std::vector<float> rgb_p = h_rgb, rgb_m = h_rgb;
    rgb_p[i] += h;
    rgb_m[i] -= h;
    auto image_p = forward_render(h_uvs, h_opacity, h_conic, rgb_p, h_num_splats_per_pixel, h_final_weight_per_pixel);
    auto image_m = forward_render(h_uvs, h_opacity, h_conic, rgb_m, h_num_splats_per_pixel, h_final_weight_per_pixel);
    double loss_p = compute_loss(image_p);
    double loss_m = compute_loss(image_m);
    float num_grad = (loss_p - loss_m) / (2.0f * h);
    EXPECT_NEAR(h_grad_rgb[i], num_grad, 1e-3);
  }

  // Cleanup
  CUDA_CHECK(cudaFree(d_uvs));
  CUDA_CHECK(cudaFree(d_opacity));
  CUDA_CHECK(cudaFree(d_conic));
  CUDA_CHECK(cudaFree(d_rgb));
  CUDA_CHECK(cudaFree(d_sorted_splats));
  CUDA_CHECK(cudaFree(d_splat_range_by_tile));
  CUDA_CHECK(cudaFree(d_num_splats_per_pixel));
  CUDA_CHECK(cudaFree(d_final_weight_per_pixel));
  CUDA_CHECK(cudaFree(d_grad_image));
  CUDA_CHECK(cudaFree(d_grad_uv));
  CUDA_CHECK(cudaFree(d_grad_opacity));
  CUDA_CHECK(cudaFree(d_grad_conic));
  CUDA_CHECK(cudaFree(d_grad_rgb));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
