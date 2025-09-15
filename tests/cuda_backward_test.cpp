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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
