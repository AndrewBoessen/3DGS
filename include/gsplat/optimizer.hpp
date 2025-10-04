// optimizer.hpp

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// Macro for checking CUDA API calls for errors.
#define CHECK_CUDA(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));               \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

/**
 * @brief Launch CUDA kernel to compute one step of Adam optimizer
 * @param[in,out]  params       A device array of parameter values
 * @param[in]      param_grads  A device array of parameter gradients
 * @param[in,out]  exp_avg      A device of exp averages
 * @param[in,out]  exp_avg_sq   A device of squared exp averages
 * @param[in]      lr           Learning rate
 * @param[in]      b1           Beta 1
 * @param[in]      b2           Beta 2
 * @param[in]      eps          Epsilon value
 * @param[in]      N            Total number of parameters
 * @param[in]      stream       The CUDA stream to execute on
 */
void adam_step(float *params, float *const param_grads, float *exp_avg, float *exp_avg_sq, const float lr,
               const float b1, const float b2, const float eps, const int N, cudaStream_t stream = 0);
