// adaptive_density.hpp

#pragma once

#include "gsplat/utils.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

int adaptive_density(const int N, const int iter, const int num_sh_coef, const ConfigParameters &config,
                     CudaDataManager &cuda, cudaStream_t stream);
