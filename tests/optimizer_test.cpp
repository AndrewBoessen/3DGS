#include "gsplat/optimizer.hpp" // The header for the code we are testing
#include <cmath>
#include <gtest/gtest.h> // Google Test framework

// A test fixture class for the Adam optimizer tests.
// This class handles the setup and teardown of memory for each test.
class AdamOptimizerTest : public ::testing::Test {
protected:
  // Per-test-suite set-up.
  // Called before the first test in this test suite.
  static void SetUpTestSuite() {
    // Ensure a CUDA device is available
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    ASSERT_GT(deviceCount, 0);
  }
  // This function is called before each test is run.
  void SetUp() override {
    // Allocate host memory
    h_params = new float[N];
    h_param_grads = new float[N];
    h_exp_avg = new float[N];
    h_exp_avg_sq = new float[N];
    h_output_params = new float[N];
    h_output_exp_avg = new float[N];
    h_output_exp_avg_sq = new float[N];

    // Initialize host data with some sample values
    for (int i = 0; i < N; ++i) {
      h_params[i] = 1.0f * i;
      h_param_grads[i] = 0.1f * (i + 1);
      h_exp_avg[i] = 0.0f;
      h_exp_avg_sq[i] = 0.0f;
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_params, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_param_grads, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_exp_avg, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_exp_avg_sq, N * sizeof(float)));

    // Copy initial data from host to device
    CHECK_CUDA(cudaMemcpy(d_params, h_params, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_param_grads, h_param_grads, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_exp_avg, h_exp_avg, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_exp_avg_sq, h_exp_avg_sq, N * sizeof(float), cudaMemcpyHostToDevice));
  }

  // This function is called after each test is run.
  void TearDown() override {
    // Free device memory
    cudaFree(d_params);
    cudaFree(d_param_grads);
    cudaFree(d_exp_avg);
    cudaFree(d_exp_avg_sq);

    // Free host memory
    delete[] h_params;
    delete[] h_param_grads;
    delete[] h_exp_avg;
    delete[] h_exp_avg_sq;
    delete[] h_output_params;
    delete[] h_output_exp_avg;
    delete[] h_output_exp_avg_sq;
  }

  // Test parameters
  const int N = 1024;
  const float lr = 0.001f;
  const float b1 = 0.9f;
  const float b2 = 0.999f;
  const float eps = 1e-8f;
  const float tolerance = 1e-6f;

  // Host pointers
  float *h_params;
  float *h_param_grads;
  float *h_exp_avg;
  float *h_exp_avg_sq;
  float *h_output_params;
  float *h_output_exp_avg;
  float *h_output_exp_avg_sq;

  // Device pointers
  float *d_params;
  float *d_param_grads;
  float *d_exp_avg;
  float *d_exp_avg_sq;
};

// Test case to verify the correctness of the Adam optimizer kernel.
// It compares the GPU result with a CPU-based implementation.
TEST_F(AdamOptimizerTest, Correctness) {
  // 1. Execute the adam_step CUDA kernel on the device
  adam_step(d_params, d_param_grads, d_exp_avg, d_exp_avg_sq, lr, b1, b2, eps, 1.0f - b1, 1.0f - b2, N);

  // Ensure the kernel has finished execution before proceeding
  CHECK_CUDA(cudaDeviceSynchronize());

  // 2. Copy the results from device memory back to host memory
  CHECK_CUDA(cudaMemcpy(h_output_params, d_params, N * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_output_exp_avg, d_exp_avg, N * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_output_exp_avg_sq, d_exp_avg_sq, N * sizeof(float), cudaMemcpyDeviceToHost));

  // 3. Calculate the expected results on the CPU
  for (int i = 0; i < N; ++i) {
    // Retrieve initial values
    float grad = h_param_grads[i];
    float param = h_params[i];
    float exp_avg = h_exp_avg[i];
    float exp_avg_sq = h_exp_avg_sq[i];

    // Perform the Adam update logic on the CPU
    exp_avg = b1 * exp_avg + (1.0f - b1) * grad;
    exp_avg_sq = b2 * exp_avg_sq + (1.0f - b2) * grad * grad;
    float m_hat = exp_avg / (1.0f - b1);
    float v_hat = exp_avg_sq / (1.0f - b2);
    float step = -lr * m_hat / (std::sqrt(v_hat) + eps);
    param += step;

    // 4. Compare the GPU result with the CPU result
    // Use ASSERT_NEAR for floating-point comparisons to allow for minor precision differences.
    ASSERT_NEAR(h_output_params[i], param, tolerance);
    ASSERT_NEAR(h_output_exp_avg[i], exp_avg, tolerance);
    ASSERT_NEAR(h_output_exp_avg_sq[i], exp_avg_sq, tolerance);
  }
}

// Main function to run all the tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
