// optimizer.hpp

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

inline constexpr float B1 = 0.9f;
inline constexpr float B2 = 0.999f;
inline constexpr float EPS = 1e-8f;

/**
 * @brief Launch CUDA kernel to compute one step of Adam optimizer
 * @param[in,out] params       A device array of parameter values
 * @param[in]     param_grads  A device array of parameter gradients
 * @param[in,out] exp_avg      A device of exp averages
 * @param[in,out] exp_avg_sq   A device of squared exp averages
 * @param[in]     lr           Learning rate
 * @param[in]     steps        Number of training steps per parameter
 * @param[in]     b1           Beta 1
 * @param[in]     b2           Beta 2
 * @param[in]     eps          Epsilon value
 * @param[in]     N            Total number of parameters
 * @param[in]     S            Stride of parameters
 * @param[in]     stream       The CUDA stream to execute on
 */
void adam_step(float *params, float *const param_grads, float *exp_avg, float *exp_avg_sq, const float lr, int *steps,
               const float b1, const float b2, const float eps, const int N, const int S, cudaStream_t stream = 0);
