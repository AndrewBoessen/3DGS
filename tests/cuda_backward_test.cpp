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

// Test for compute_conic_backward
TEST_F(CudaBackwardKernelTest, ConicBackward) {
  const int N = 1;
  const float h = 1e-4;

  // Host data - Jacobian matrix (2x3, stored as 6 elements)
  std::vector<float> h_J = {100.0, 0.0, -33.33, 0.0, 120.0, -80.0};
  // 3D covariance (upper triangular, 6 elements)
  std::vector<float> h_sigma = {0.5, 0.1, 0.2, 0.6, 0.15, 0.7};
  // Camera extrinsic matrix (3x4)
  std::vector<float> h_T = {0.8, -0.6, 0.0, 0.1, 0.6, 0.8, 0.0, 0.2, 0.0, 0.0, 1.0, 0.3};
  // Gradient from conic (upper triangular, 3 elements)
  std::vector<float> h_conic_grad_out = {0.1, 0.2, 0.3};
  std::vector<float> h_J_grad_in(N * 6);
  std::vector<float> h_sigma_grad_in(N * 6);

  // Device data
  float *d_J = device_alloc<float>(N * 6);
  float *d_sigma = device_alloc<float>(N * 6);
  float *d_T = device_alloc<float>(12);
  float *d_conic_grad_out = device_alloc<float>(N * 3);
  float *d_J_grad_in = device_alloc<float>(N * 6);
  float *d_sigma_grad_in = device_alloc<float>(N * 6);

  CUDA_CHECK(cudaMemcpy(d_J, h_J.data(), N * 6 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sigma, h_sigma.data(), N * 6 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_T, h_T.data(), 12 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_conic_grad_out, h_conic_grad_out.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));

  // Run kernel
  compute_conic_backward(d_J, d_sigma, d_T, d_conic_grad_out, N, d_J_grad_in, d_sigma_grad_in);

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_J_grad_in.data(), d_J_grad_in, N * 6 * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_sigma_grad_in.data(), d_sigma_grad_in, N * 6 * sizeof(float), cudaMemcpyDeviceToHost));

  // Numerical gradient check
  auto forward_conic = [&](const std::vector<float> &J, const std::vector<float> &sigma, const std::vector<float> &T) {
    // Extract rotation part of T (first 3x3)
    float R[9] = {T[0], T[1], T[2], T[4], T[5], T[6], T[8], T[9], T[10]};

    // Transform sigma from world to camera: sigma_c = R * sigma * R^T
    // First compute sigma_full (3x3 from upper triangular)
    float sigma_full[9] = {sigma[0], sigma[1], sigma[2], sigma[1], sigma[3], sigma[4], sigma[2], sigma[4], sigma[5]};

    // R * sigma
    float RS[9];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        RS[i * 3 + j] = 0;
        for (int k = 0; k < 3; ++k) {
          RS[i * 3 + j] += R[i * 3 + k] * sigma_full[k * 3 + j];
        }
      }
    }

    // (R * sigma) * R^T
    float sigma_c[9];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        sigma_c[i * 3 + j] = 0;
        for (int k = 0; k < 3; ++k) {
          sigma_c[i * 3 + j] += RS[i * 3 + k] * R[j * 3 + k];
        }
      }
    }

    // Project to 2D: conic = J * sigma_c * J^T
    // J is 2x3, sigma_c is 3x3
    float J_mat[6] = {J[0], J[1], J[2], J[3], J[4], J[5]};

    // J * sigma_c
    float JS[6];
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        JS[i * 3 + j] = 0;
        for (int k = 0; k < 3; ++k) {
          JS[i * 3 + j] += J_mat[i * 3 + k] * sigma_c[k * 3 + j];
        }
      }
    }

    // (J * sigma_c) * J^T
    float conic_full[4];
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        conic_full[i * 2 + j] = 0;
        for (int k = 0; k < 3; ++k) {
          conic_full[i * 2 + j] += JS[i * 3 + k] * J_mat[j * 3 + k];
        }
      }
    }

    // Return upper triangular part
    std::vector<float> conic(3);
    conic[0] = conic_full[0]; // (0,0)
    conic[1] = conic_full[1]; // (0,1)
    conic[2] = conic_full[3]; // (1,1)
    return conic;
  };

  // Check grad w.r.t J
  for (int i = 0; i < 6; ++i) {
    std::vector<float> J_p = h_J;
    J_p[i] += h;
    std::vector<float> J_m = h_J;
    J_m[i] -= h;
    auto conic_p = forward_conic(J_p, h_sigma, h_T);
    auto conic_m = forward_conic(J_m, h_sigma, h_T);
    float numerical_grad = 0;
    for (int j = 0; j < 3; ++j)
      numerical_grad += (conic_p[j] - conic_m[j]) / (2 * h) * h_conic_grad_out[j];
    ASSERT_NEAR(h_J_grad_in[i], numerical_grad, 1e-2);
  }

  // Check grad w.r.t sigma
  for (int i = 0; i < 6; ++i) {
    std::vector<float> sigma_p = h_sigma;
    sigma_p[i] += h;
    std::vector<float> sigma_m = h_sigma;
    sigma_m[i] -= h;
    auto conic_p = forward_conic(h_J, sigma_p, h_T);
    auto conic_m = forward_conic(h_J, sigma_m, h_T);
    float numerical_grad = 0;
    for (int j = 0; j < 3; ++j)
      numerical_grad += (conic_p[j] - conic_m[j]) / (2 * h) * h_conic_grad_out[j];
    ASSERT_NEAR(h_sigma_grad_in[i], numerical_grad, 1e-2);
  }

  CUDA_CHECK(cudaFree(d_J));
  CUDA_CHECK(cudaFree(d_sigma));
  CUDA_CHECK(cudaFree(d_T));
  CUDA_CHECK(cudaFree(d_conic_grad_out));
  CUDA_CHECK(cudaFree(d_J_grad_in));
  CUDA_CHECK(cudaFree(d_sigma_grad_in));
}

// Test for compute_sigma_backward
TEST_F(CudaBackwardKernelTest, SigmaBackward) {
  const int N = 1;
  const float h = 1e-4;

  std::vector<float> h_quat = {0.9, 0.1, 0.2, 0.3};
  std::vector<float> h_scale = {0.5, 0.6, 0.7};
  std::vector<float> h_sigma_grad_in = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::vector<float> h_quat_grad_out(N * 4);
  std::vector<float> h_scale_grad_out(N * 3);

  auto d_quat = device_alloc<float>(N * 4);
  auto d_scale = device_alloc<float>(N * 3);
  auto d_sigma_grad_in = device_alloc<float>(N * 6);
  auto d_quat_grad_out = device_alloc<float>(N * 4);
  auto d_scale_grad_out = device_alloc<float>(N * 3);

  CUDA_CHECK(cudaMemcpy(d_quat, h_quat.data(), N * 4 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale, h_scale.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sigma_grad_in, h_sigma_grad_in.data(), N * 6 * sizeof(float), cudaMemcpyHostToDevice));

  compute_sigma_backward(d_quat, d_scale, d_sigma_grad_in, N, d_quat_grad_out, d_scale_grad_out);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_quat_grad_out.data(), d_quat_grad_out, N * 4 * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_scale_grad_out.data(), d_scale_grad_out, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  auto forward_sigma = [](const std::vector<float> &quat, const std::vector<float> &scale) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    float norm = std::sqrt(w * w + x * x + y * y + z * z);
    w /= norm;
    x /= norm;
    y /= norm;
    z /= norm;

    float R[] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
                 2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                 2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)};

    // The forward pass is based on R * S^2 * R^T
    float S_sq[] = {std::exp(scale[0]) * std::exp(scale[0]), std::exp(scale[1]) * std::exp(scale[1]),
                    std::exp(scale[2]) * std::exp(scale[2])};

    // RM = R * S^2
    float RM[9];
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        RM[r * 3 + c] = R[r * 3 + c] * S_sq[c];

    // sigma = RM * R^T
    std::vector<float> sigma(6);
    sigma[0] = RM[0] * R[0] + RM[1] * R[1] + RM[2] * R[2]; // xx
    sigma[1] = RM[0] * R[3] + RM[1] * R[4] + RM[2] * R[5]; // xy
    sigma[2] = RM[0] * R[6] + RM[1] * R[7] + RM[2] * R[8]; // xz
    sigma[3] = RM[3] * R[3] + RM[4] * R[4] + RM[5] * R[5]; // yy
    sigma[4] = RM[3] * R[6] + RM[4] * R[7] + RM[5] * R[8]; // yz
    sigma[5] = RM[6] * R[6] + RM[7] * R[7] + RM[8] * R[8]; // zz
    return sigma;
  };

  for (int i = 0; i < 4; ++i) {
    std::vector<float> quat_p = h_quat;
    quat_p[i] += h;
    std::vector<float> quat_m = h_quat;
    quat_m[i] -= h;
    auto sigma_p = forward_sigma(quat_p, h_scale);
    auto sigma_m = forward_sigma(quat_m, h_scale);
    float numerical_grad = 0;
    for (int j = 0; j < 6; ++j)
      numerical_grad += (sigma_p[j] - sigma_m[j]) / (2 * h) * h_sigma_grad_in[j];
    ASSERT_NEAR(h_quat_grad_out[i], numerical_grad, 1e-1);
  }

  for (int i = 0; i < 3; ++i) {
    std::vector<float> scale_p = h_scale;
    scale_p[i] += h;
    std::vector<float> scale_m = h_scale;
    scale_m[i] -= h;
    auto sigma_p = forward_sigma(h_quat, scale_p);
    auto sigma_m = forward_sigma(h_quat, scale_m);
    float numerical_grad = 0;
    for (int j = 0; j < 6; ++j)
      numerical_grad += (sigma_p[j] - sigma_m[j]) / (2 * h) * h_sigma_grad_in[j];
    ASSERT_NEAR(h_scale_grad_out[i], numerical_grad, 1e-1);
  }

  CUDA_CHECK(cudaFree(d_quat));
  CUDA_CHECK(cudaFree(d_scale));
  CUDA_CHECK(cudaFree(d_sigma_grad_in));
  CUDA_CHECK(cudaFree(d_quat_grad_out));
  CUDA_CHECK(cudaFree(d_scale_grad_out));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
