// optimizer.hpp

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

inline constexpr float B1 = 0.9f;
inline constexpr float B2 = 0.999f;
inline constexpr float EPS = 1e-8f;

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
 * @brief Filter optimzier moment vactors based on mask
 * @param[in]  N           Number of parameters
 * @param[in]  S           Size of stride
 * @param[in]  d_mask      Binary mask
 * @param[in]  d_m         Original first moment vector
 * @param[in]  d_v         Original second moment vector
 * @param[out] d_m_culled  Filtered m
 * @param[out] d_v_culled  Filtered v
 * @param[in]  stream      The CUDA stream to execute on
 */
void filter_moment_vectors(const int N, const int S, const bool *d_mask, const float *d_m, const float *d_v,
                           float *d_m_culled, float *d_v_culled, cudaStream_t stream = 0);

/**
 * @brief Launch CUDA kernel to compute one step of Adam optimizer
 * @param[in,out] params       A device array of parameter values
 * @param[in]     param_grads  A device array of parameter gradients
 * @param[in,out] exp_avg      A device of exp averages
 * @param[in,out] exp_avg_sq   A device of squared exp averages
 * @param[in]     lr           Learning rate
 * @param[in]     b1           Beta 1
 * @param[in]     b2           Beta 2
 * @param[in]     eps          Epsilon value
 * @param[in]     b1_t_corr    Beta 1 bias correction term
 * @param[in]     b2_t_corr    Beta 2 bias correction term
 * @param[in]     N            Total number of parameters
 * @param[in]     stream       The CUDA stream to execute on
 */
void adam_step(float *params, float *const param_grads, float *exp_avg, float *exp_avg_sq, const float lr,
               const float b1, const float b2, const float eps, const float b1_t_corr, const float b2_t_corr,
               const int N, cudaStream_t stream = 0);
