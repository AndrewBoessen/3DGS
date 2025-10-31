// adaptive_density.cu

#include "checks.cuh"
#include "gsplat_cuda/adaptive_density.cuh"
#include <curand_kernel.h>
#include <thrust/device_vector.h>

__global__ void setup_states(curandState *state, unsigned long seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, idx, 0, &state[idx]);
}

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
    const int write_base = write_ids[i] * 2;

    const float3 xyz = {xyz_in[i * 3 + 0], xyz_in[i * 3 + 1], xyz_in[i * 3 + 2]};
    const float3 rgb = {rgb_in[i * 3 + 0], rgb_in[i * 3 + 1], rgb_in[i * 3 + 2]};
    const float op = op_in[i];
    const float3 scale = {scale_in[i * 3 + 0], scale_in[i * 3 + 1], scale_in[i * 3 + 2]};
    const float4 quat = {quat_in[i * 4 + 0], quat_in[i * 4 + 1], quat_in[i * 4 + 2], quat_in[i * 4 + 3]};

    // move cloned in xyz grad direction
    const int accum_count = accum_dur[i];
    const float3 xyz_grad_accum = {xyz_grad[i * 3 + 0] / accum_count, xyz_grad[i * 3 + 1] / accum_count,
                                   xyz_grad[i * 3 + 2] / accum_count};

    const float3 xyz_new = {xyz.x - xyz_grad_accum.x * 0.01f, xyz.y - xyz_grad_accum.y * 0.01f,
                            xyz.z - xyz_grad_accum.z * 0.01f};

    // write cloned parameters
    for (int j = 0; j < 2; j++) {
      if (j == 0) {
        xyz_out[(write_base + j) * 3 + 0] = xyz_new.x;
        xyz_out[(write_base + j) * 3 + 1] = xyz_new.y;
        xyz_out[(write_base + j) * 3 + 2] = xyz_new.z;
      } else {
        xyz_out[(write_base + j) * 3 + 0] = xyz.x;
        xyz_out[(write_base + j) * 3 + 1] = xyz.y;
        xyz_out[(write_base + j) * 3 + 2] = xyz.z;
      }

      rgb_out[(write_base + j) * 3 + 0] = rgb.x;
      rgb_out[(write_base + j) * 3 + 1] = rgb.y;
      rgb_out[(write_base + j) * 3 + 2] = rgb.z;

      op_out[(write_base + j)] = op;

      scale_out[(write_base + j) * 3 + 0] = scale.x;
      scale_out[(write_base + j) * 3 + 1] = scale.y;
      scale_out[(write_base + j) * 3 + 2] = scale.z;

      quat_out[(write_base + j) * 4 + 0] = quat.x;
      quat_out[(write_base + j) * 4 + 1] = quat.y;
      quat_out[(write_base + j) * 4 + 2] = quat.z;
      quat_out[(write_base + j) * 4 + 3] = quat.w;

      // handle SH coefficients
      if (num_sh_coef > 0) {
        for (int k = 0; k < num_sh_coef; k++) {
#pragma unroll
          for (int l = 0; l < 3; l++) {
            const int sh_write_id = ((write_base + j) * num_sh_coef * 3) + (k * 3) + l;
            const int sh_read_id = ((i + j) * num_sh_coef * 3) + (k * 3) + l;
            sh_out[sh_write_id] = sh_in[sh_read_id];
          }
        }
      }
    }
  }
}

void __global__ split_gaussians_kernel(const int N, const float scale_factor, const int num_sh_coef,
                                       const bool *__restrict__ mask, const int *__restrict__ write_ids,
                                       const float *__restrict__ xyz_in, const float *__restrict__ rgb_in,
                                       const float *__restrict__ op_in, const float *__restrict__ scale_in,
                                       const float *__restrict__ quat_in, const float *__restrict__ sh_in,
                                       float *xyz_out, float *rgb_out, float *op_out, float *scale_out, float *quat_out,
                                       float *sh_out, curandState *state) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= N)
    return;

  const bool is_split = mask[i];

  if (is_split) {
    const int write_base = write_ids[i] * 2;

    const float3 xyz = {xyz_in[i * 3 + 0], xyz_in[i * 3 + 1], xyz_in[i * 3 + 2]};
    const float3 rgb = {rgb_in[i * 3 + 0], rgb_in[i * 3 + 1], rgb_in[i * 3 + 2]};
    const float op = op_in[i];
    const float3 scale = {scale_in[i * 3 + 0], scale_in[i * 3 + 1], scale_in[i * 3 + 2]};
    const float4 quat = {quat_in[i * 4 + 0], quat_in[i * 4 + 1], quat_in[i * 4 + 2], quat_in[i * 4 + 3]};

    const float3 exp_scale = {expf(scale.x), expf(scale.y), expf(scale.z)};

    const float inv_norm = rsqrtf(quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w);
    // Convert float4 to quaternion form
    const float w = quat.x * inv_norm;
    const float x = quat.y * inv_norm;
    const float y = quat.z * inv_norm;
    const float z = quat.w * inv_norm;

    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float wx = w * x;
    const float wy = w * y;
    const float wz = w * z;

    // R (rotation matrix)
    float r00 = 1.0f - 2.0f * (y2 + z2);
    float r01 = 2.0f * (xy - wz);
    float r02 = 2.0f * (xz + wy);
    float r10 = 2.0f * (xy + wz);
    float r11 = 1.0f - 2.0f * (x2 + z2);
    float r12 = 2.0f * (yz - wx);
    float r20 = 2.0f * (xz - wy);
    float r21 = 2.0f * (yz + wx);
    float r22 = 1.0f - 2.0f * (x2 + y2);

    // Sample from Gaussian to find new centers
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const float v_x = curand_normal(&state[i]) * exp_scale.x;
      const float v_y = curand_normal(&state[i]) * exp_scale.y;
      const float v_z = curand_normal(&state[i]) * exp_scale.z;

      const float rot_v_x = v_x * r00 + v_y * r01 + v_z * r02;
      const float rot_v_y = v_x * r10 + v_y * r11 + v_z * r12;
      const float rot_v_z = v_x * r20 + v_y * r21 + v_z * r22;

      // Write new parameters
      xyz_out[(write_base + j) * 3 + 0] = xyz.x + rot_v_x;
      xyz_out[(write_base + j) * 3 + 1] = xyz.y + rot_v_y;
      xyz_out[(write_base + j) * 3 + 2] = xyz.z + rot_v_z;

      rgb_out[(write_base + j) * 3 + 0] = rgb.x;
      rgb_out[(write_base + j) * 3 + 1] = rgb.y;
      rgb_out[(write_base + j) * 3 + 2] = rgb.z;

      op_out[write_base + j] = op;

      scale_out[(write_base + j) * 3 + 0] = logf(exp_scale.x / scale_factor);
      scale_out[(write_base + j) * 3 + 1] = logf(exp_scale.y / scale_factor);
      scale_out[(write_base + j) * 3 + 2] = logf(exp_scale.z / scale_factor);

      quat_out[(write_base + j) * 4 + 0] = quat.x;
      quat_out[(write_base + j) * 4 + 1] = quat.y;
      quat_out[(write_base + j) * 4 + 2] = quat.z;
      quat_out[(write_base + j) * 4 + 3] = quat.w;

      // handle SH coefficients
      if (num_sh_coef > 0) {
        for (int k = 0; k < num_sh_coef; k++) {
#pragma unroll
          for (int l = 0; l < 3; l++) {
            const int sh_write_id = ((write_base + j) * num_sh_coef * 3) + (k * 3) + l;
            const int sh_read_id = ((i + j) * num_sh_coef * 3) + (k * 3) + l;
            sh_out[sh_write_id] = sh_in[sh_read_id];
          }
        }
      }
    }
  }
}

void clone_gaussians(const int N, const int num_sh_coef, const bool *mask, const int *write_ids, const float *xyz_grad,
                     const int *accum_dur, const float *xyz_in, const float *rgb_in, const float *op_in,
                     const float *scale_in, const float *quat_in, const float *sh_in, float *xyz_out, float *rgb_out,
                     float *op_out, float *scale_out, float *quat_out, float *sh_out, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(mask);
  ASSERT_DEVICE_POINTER(write_ids);
  ASSERT_DEVICE_POINTER(xyz_grad);
  ASSERT_DEVICE_POINTER(accum_dur);

  if (N == 0)
    return;

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  clone_gaussians_kernel<<<blocks, threads, 0, stream>>>(N, num_sh_coef, mask, write_ids, xyz_grad, accum_dur, xyz_in,
                                                         rgb_in, op_in, scale_in, quat_in, sh_in, xyz_out, rgb_out,
                                                         op_out, scale_out, quat_out, sh_out);
}

void split_gaussians(const int N, const float scale_factor, const int num_sh_coef, const bool *mask,
                     const int *write_ids, const float *xyz_in, const float *rgb_in, const float *op_in,
                     const float *scale_in, const float *quat_in, const float *sh_in, float *xyz_out, float *rgb_out,
                     float *op_out, float *scale_out, float *quat_out, float *sh_out, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(mask);
  ASSERT_DEVICE_POINTER(write_ids);

  if (N == 0)
    return;

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  thrust::device_vector<curandState> states(threads * blocks);

  setup_states<<<blocks, threads, 0, stream>>>(thrust::raw_pointer_cast(states.data()), (uint64_t)time(NULL));

  split_gaussians_kernel<<<blocks, threads, 0, stream>>>(
      N, scale_factor, num_sh_coef, mask, write_ids, xyz_in, rgb_in, op_in, scale_in, quat_in, sh_in, xyz_out, rgb_out,
      op_out, scale_out, quat_out, sh_out, thrust::raw_pointer_cast(states.data()));
}
