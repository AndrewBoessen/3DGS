#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include "gsplat/cuda_backward.hpp"

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

// Test for camera_intrinsic_projection_backward
TEST_F(CudaBackwardKernelTest, CameraIntrinsicProjectionBackward) {
  const int N = 2;
  const float h = 1e-4;

  // Host data
  std::vector<float> h_xyz_c = {1.0, 2.0, 3.0, -1.0, -2.0, 4.0};
  std::vector<float> h_K = {100.0, 0.0, 160.0, 0.0, 120.0, 120.0, 0.0, 0.0, 1.0}; // fx, cx, fy, cy
  std::vector<float> h_uv_grad_out = {0.1, 0.2, 0.3, 0.4};
  std::vector<float> h_xyz_c_grad_in(N * 3);
  std::vector<float> h_K_grad_in(4);

  // Device data
  float *d_xyz_c = device_alloc<float>(N * 3);
  float *d_K = device_alloc<float>(9);
  float *d_uv_grad_out = device_alloc<float>(N * 2);
  float *d_xyz_c_grad_in = device_alloc<float>(N * 3);

  CUDA_CHECK(cudaMemcpy(d_xyz_c, h_xyz_c.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), 9 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_uv_grad_out, h_uv_grad_out.data(), N * 2 * sizeof(float), cudaMemcpyHostToDevice));

  // Run kernel
  camera_intrinsic_projection_backward(d_xyz_c, d_K, d_uv_grad_out, N, d_xyz_c_grad_in);

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_xyz_c_grad_in.data(), d_xyz_c_grad_in, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  // Numerical gradient check
  auto forward_proj = [&](const std::vector<float> &xyz_c, const std::vector<float> &K) {
    std::vector<float> uv(N * 2);
    for (int i = 0; i < N; ++i) {
      uv[i * 2 + 0] = K[0] * xyz_c[i * 3 + 0] / xyz_c[i * 3 + 2] + K[2];
      uv[i * 2 + 1] = K[5] * xyz_c[i * 3 + 1] / xyz_c[i * 3 + 2] + K[5];
    }
    return uv;
  };

  // Check grad w.r.t xyz_c
  for (int i = 0; i < N * 3; ++i) {
    std::vector<float> xyz_c_p = h_xyz_c;
    xyz_c_p[i] += h;
    std::vector<float> xyz_c_m = h_xyz_c;
    xyz_c_m[i] -= h;
    auto uv_p = forward_proj(xyz_c_p, h_K);
    auto uv_m = forward_proj(xyz_c_m, h_K);
    float numerical_grad = 0;
    for (int j = 0; j < N * 2; ++j)
      numerical_grad += (uv_p[j] - uv_m[j]) / (2 * h) * h_uv_grad_out[j];
    ASSERT_NEAR(h_xyz_c_grad_in[i], numerical_grad, 1e-1);
  }

  CUDA_CHECK(cudaFree(d_xyz_c));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_uv_grad_out));
  CUDA_CHECK(cudaFree(d_xyz_c_grad_in));
}

// Test for camera_extrinsic_projection_backward
TEST_F(CudaBackwardKernelTest, CameraExtrinsicProjectionBackward) {
  const int N = 1;
  const float h = 1e-4;

  // Host data
  std::vector<float> h_xyz_w = {1.0, 2.0, 3.0};
  std::vector<float> h_T = {0.8, -0.6, 0.0, 0.1, 0.6, 0.8, 0.0, 0.2, 0.0, 0.0, 1.0, 0.3};
  std::vector<float> h_xyz_c_grad_in = {0.1, 0.2, 0.3};
  std::vector<float> h_xyz_w_grad_in(N * 3);
  std::vector<float> h_T_grad_in(12);

  // Device data
  auto d_xyz_w = device_alloc<float>(N * 3);
  auto d_T = device_alloc<float>(12);
  auto d_xyz_c_grad_in = device_alloc<float>(N * 3);
  auto d_xyz_w_grad_in = device_alloc<float>(N * 3);

  CUDA_CHECK(cudaMemcpy(d_xyz_w, h_xyz_w.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_T, h_T.data(), 12 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_xyz_c_grad_in, h_xyz_c_grad_in.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));

  camera_extrinsic_projection_backward(d_xyz_w, d_T, d_xyz_c_grad_in, N, d_xyz_w_grad_in);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_xyz_w_grad_in.data(), d_xyz_w_grad_in, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  auto forward_ext = [&](const std::vector<float> &xyz_w, const std::vector<float> &T) {
    std::vector<float> xyz_c(N * 3);
    for (int i = 0; i < N; ++i) {
      xyz_c[i * 3 + 0] = T[0] * xyz_w[i * 3 + 0] + T[1] * xyz_w[i * 3 + 1] + T[2] * xyz_w[i * 3 + 2] + T[3];
      xyz_c[i * 3 + 1] = T[4] * xyz_w[i * 3 + 0] + T[5] * xyz_w[i * 3 + 1] + T[6] * xyz_w[i * 3 + 2] + T[7];
      xyz_c[i * 3 + 2] = T[8] * xyz_w[i * 3 + 0] + T[9] * xyz_w[i * 3 + 1] + T[10] * xyz_w[i * 3 + 2] + T[11];
    }
    return xyz_c;
  };

  for (int i = 0; i < N * 3; ++i) {
    std::vector<float> xyz_w_p = h_xyz_w;
    xyz_w_p[i] += h;
    std::vector<float> xyz_w_m = h_xyz_w;
    xyz_w_m[i] -= h;
    auto xyz_c_p = forward_ext(xyz_w_p, h_T);
    auto xyz_c_m = forward_ext(xyz_w_m, h_T);
    float numerical_grad = 0;
    for (int j = 0; j < N * 3; ++j)
      numerical_grad += (xyz_c_p[j] - xyz_c_m[j]) / (2 * h) * h_xyz_c_grad_in[j];
    ASSERT_NEAR(h_xyz_w_grad_in[i], numerical_grad, 1e-1);
  }

  CUDA_CHECK(cudaFree(d_xyz_w));
  CUDA_CHECK(cudaFree(d_T));
  CUDA_CHECK(cudaFree(d_xyz_c_grad_in));
  CUDA_CHECK(cudaFree(d_xyz_w_grad_in));
}

// Test for compute_projection_jacobian_backward
TEST_F(CudaBackwardKernelTest, ProjectionJacobianBackward) {
  const int N = 2;
  const float h = 1e-4;

  // Host data
  std::vector<float> h_xyz_c = {1.0, 2.0, 3.0, -1.0, -2.0, 4.0};
  std::vector<float> h_K = {100.0, 0.0, 160.0, 0.0, 120.0, 120.0, 0.0, 0.0, 1.0};
  std::vector<float> h_J_grad_in = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
  std::vector<float> h_xyz_c_grad_out(N * 3);

  // Device data
  float *d_xyz_c = device_alloc<float>(N * 3);
  float *d_K = device_alloc<float>(9);
  float *d_J_grad_in = device_alloc<float>(N * 6);
  float *d_xyz_c_grad_out = device_alloc<float>(N * 3);

  CUDA_CHECK(cudaMemcpy(d_xyz_c, h_xyz_c.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), 9 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_J_grad_in, h_J_grad_in.data(), N * 6 * sizeof(float), cudaMemcpyHostToDevice));

  // Run kernel
  compute_projection_jacobian_backward(d_xyz_c, d_K, d_J_grad_in, N, d_xyz_c_grad_out);

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_xyz_c_grad_out.data(), d_xyz_c_grad_out, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  // Numerical gradient check
  auto forward_jacobian = [&](const std::vector<float> &xyz_c, const std::vector<float> &K) {
    std::vector<float> J(N * 6);
    for (int i = 0; i < N; ++i) {
      float x = xyz_c[i * 3 + 0];
      float y = xyz_c[i * 3 + 1];
      float z = xyz_c[i * 3 + 2];
      float fx = K[0], fy = K[4];
      float z_inv = 1.0f / z;
      float z_inv2 = z_inv * z_inv;

      // Jacobian: du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz
      J[i * 6 + 0] = fx * z_inv;       // du/dx
      J[i * 6 + 1] = 0.0f;             // du/dy
      J[i * 6 + 2] = -fx * x * z_inv2; // du/dz
      J[i * 6 + 3] = 0.0f;             // dv/dx
      J[i * 6 + 4] = fy * z_inv;       // dv/dy
      J[i * 6 + 5] = -fy * y * z_inv2; // dv/dz
    }
    return J;
  };

  // Check grad w.r.t xyz_c
  for (int i = 0; i < N * 3; ++i) {
    std::vector<float> xyz_c_p = h_xyz_c;
    xyz_c_p[i] += h;
    std::vector<float> xyz_c_m = h_xyz_c;
    xyz_c_m[i] -= h;
    auto J_p = forward_jacobian(xyz_c_p, h_K);
    auto J_m = forward_jacobian(xyz_c_m, h_K);
    float numerical_grad = 0;
    for (int j = 0; j < N * 6; ++j)
      numerical_grad += (J_p[j] - J_m[j]) / (2 * h) * h_J_grad_in[j];
    ASSERT_NEAR(h_xyz_c_grad_out[i], numerical_grad, 1e-1);
  }

  CUDA_CHECK(cudaFree(d_xyz_c));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_J_grad_in));
  CUDA_CHECK(cudaFree(d_xyz_c_grad_out));
}

TEST_F(CudaBackwardKernelTest, SigmaBackward) {
  const int N = 2;
  const float h = 1e-4;

  // Host data
  std::vector<float> h_q = {1.0,        0.1,        0.2, 0.3,  // Gaussian 1
                            0.70710678, 0.70710678, 0.0, 0.0}; // Gaussian 2: 90 deg rot around X
  std::vector<float> h_s = {0.1, 0.2, 0.3, -0.1, -0.2, -0.3};
  std::vector<float> h_dSigma_in = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6};
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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
