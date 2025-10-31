// adaptive_density.cu

#include "checks.cuh"
#include "gsplat_cuda/adaptive_density.cuh"
#include <thrust/device_vector.h>

void __global__ clone_gaussians_kernel(const int N, const int num_sh_coef, const bool *__restrict__ mask,
                                       const int *__restrict__ write_ids, const float *__restrict__ xyz_grad,
                                       const int *__restrict__ accum_dur, const float *__restrict__ xyz_in,
                                       const float *__restrict__ rgb_in, const float *__restrict__ op_in,
                                       const float *__restrict__ scale_in, const float *__restrict__ quat_in,
                                       const float *__restrict__ sh_in, float *xyz_out, float *rgb_out, float *op_out,
                                       float *scale_out, float *quat_out, float *sh_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= N)
    return;

  const bool is_cloned = mask[i];

  if (is_cloned) {
    const int write_id = write_ids[i];

    float3 xyz = {xyz_in[i * 3 + 0], xyz_in[i * 3 + 1], xyz_in[i * 3 + 2]};
    const float3 rgb = {rgb_in[i * 3 + 0], rgb_in[i * 3 + 1], rgb_in[i * 3 + 2]};
    const float op = op_in[i];
    const float3 scale = {scale_in[i * 3 + 0], scale_in[i * 3 + 1], scale_in[i * 3 + 2]};
    const float4 quat = {quat_in[i * 4 + 0], quat_in[i * 4 + 1], quat_in[i * 4 + 2], quat_in[i * 4 + 3]};

    // move cloned in xyz grad direction
    const int accum_count = accum_dur[i];
    const float3 xyz_grad_accum = {xyz_grad[i * 3 + 0] / accum_count, xyz_grad[i * 3 + 1] / accum_count,
                                   xyz_grad[i * 3 + 2] / accum_count};

    xyz.x = xyz_grad_accum.x * 0.01f + xyz.x;
    xyz.y = xyz_grad_accum.y * 0.01f + xyz.y;
    xyz.z = xyz_grad_accum.z * 0.01f + xyz.z;

    // write cloned parameters
    xyz_out[write_id * 3 + 0] = xyz.x;
    xyz_out[write_id * 3 + 1] = xyz.y;
    xyz_out[write_id * 3 + 2] = xyz.z;

    rgb_out[write_id * 3 + 0] = rgb.x;
    rgb_out[write_id * 3 + 1] = rgb.y;
    rgb_out[write_id * 3 + 2] = rgb.z;

    op_out[write_id] = op;

    scale_out[write_id * 3 + 0] = scale.x;
    scale_out[write_id * 3 + 1] = scale.y;
    scale_out[write_id * 3 + 2] = scale.z;

    quat_out[write_id * 4 + 0] = quat.x;
    quat_out[write_id * 4 + 1] = quat.y;
    quat_out[write_id * 4 + 2] = quat.z;
    quat_out[write_id * 4 + 3] = quat.w;

    // handle SH coefficients
    if (num_sh_coef > 0) {
      for (int i = 0; i < num_sh_coef; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
          const int sh_write_id = (write_id * num_sh_coef * 3) + (i * 3) + j;
          sh_out[sh_write_id] = sh_in[sh_write_id];
        }
      }
    }
  }
}

void __global__ split_gaussians_kernel(const int N, const int num_sh_coef, const bool *__restrict__ mask,
                                       const int *__restrict__ write_ids, const float *__restrict__ xyz_in,
                                       const float *__restrict__ rgb_in, const float *__restrict__ op_in,
                                       const float *__restrict__ scale_in, const float *__restrict__ quat_in,
                                       const float *__restrict__ sh_in, const float *xyz_out, const float *rgb_out,
                                       const float *op_out, const float *scale_out, const float *quat_out,
                                       const float *sh_out) {}

void clone() {}
void split() {}
