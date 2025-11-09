// optimizer.cu

#include "checks.cuh"
#include "gsplat_cuda/optimizer.cuh"

__global__ void adam_kernel(const int N, const int S, float *__restrict__ param, const float *__restrict__ param_grad,
                            float *__restrict__ exp_avg, float *__restrict__ exp_avg_sq, const float lr, const float b1,
                            const float b2, const float eps, const float bias1, const float bias2) {
  const int p_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (p_idx >= N * S)
    return;

  float register_param_grad = param_grad[p_idx];
  float register_exp_avg = exp_avg[p_idx];
  float register_exp_avg_sq = exp_avg_sq[p_idx];
  register_exp_avg = b1 * register_exp_avg + (1.0f - b1) * register_param_grad;
  register_exp_avg_sq = b2 * register_exp_avg_sq + (1.0f - b2) * register_param_grad * register_param_grad;
  // Compute bias-corrected estimates using pre-calculated values
  float m_hat = register_exp_avg / bias1;
  float v_hat = register_exp_avg_sq / bias2;
  float step = -lr * m_hat / (sqrt(v_hat) + eps);

  param[p_idx] += step;
  exp_avg[p_idx] = register_exp_avg;
  exp_avg_sq[p_idx] = register_exp_avg_sq;
};

void adam_step(float *params, float *const param_grads, float *exp_avg, float *exp_avg_sq, const float lr,
               const float b1, const float b2, const float eps, const float bias1, const float bias2, const int N,
               const int S, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(params);
  ASSERT_DEVICE_POINTER(param_grads);
  ASSERT_DEVICE_POINTER(exp_avg);
  ASSERT_DEVICE_POINTER(exp_avg_sq);

  dim3 threads(256);
  dim3 blocks((N * S + threads.x - 1) / threads.x);

  adam_kernel<<<blocks, threads, 0, stream>>>(N, S, params, param_grads, exp_avg, exp_avg_sq, lr, b1, b2, eps, bias1,
                                              bias2);
}
