// adaptive_density.cu

#include "checks.cuh"
#include "gsplat/adaptive_density.hpp"
#include <cmath>
#include <ctime>
#include <cub/cub.cuh>
#include <curand_kernel.h>

__device__ __forceinline__ int get_strided_write_index(const int stride, const bool write, const int lane,
                                                       const unsigned int active_mask, int *global_index) {
  // Get mask of threads that want to write
  unsigned mask = __ballot_sync(active_mask, write);
  // no threads in warp will write
  if (mask == 0) {
    return -1;
  }
  // Count how many threads before want to write
  int prefix = __popc(mask & ((1u << lane) - 1));

  // First active thread in warp gets block of slots
  int base_slot = 0;
  int leader_lane = __ffs(mask) - 1;
  if (lane == leader_lane) {
    base_slot = atomicAdd(global_index, __popc(mask) * stride);
  }

  // Broadcast base slot to all threads
  base_slot = __shfl_sync(active_mask, base_slot, leader_lane);

  return write ? (base_slot + prefix * stride) : -1;
}

__device__ __forceinline__ bool keep_test(const float opacity, const float op_threshold, const float scale_max,
                                          const float extent) {
  const float op_param_threshold = __logf(op_threshold) - __logf(1.0f - op_threshold);
  return !(opacity < op_param_threshold || scale_max > 0.1f * extent);
}

__global__ void setup_states(curandState *state, unsigned long seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, idx, 0, &state[idx]);
}

__global__ void fused_adaptive_density_kernel(
    const int N, const int max_gaussians, const bool use_delete, const bool use_clone, const bool use_split,
    const float op_threshold, const float extent, const float percent_dense, const float grad_threshold,
    const int num_split_samples, const float split_scale_factor, const float *__restrict__ uv_grad_accum,
    const float *__restrict__ xyz_grad_accum, const int *__restrict__ grad_accum_count, float *xyz, float *rgb,
    float *sh, float *opacity, float *scale, float *quaternion, float *m_xyz, float *v_xyz, float *m_rgb, float *v_rgb,
    float *m_op, float *v_op, float *m_scale, float *v_scale, float *m_quat, float *v_quat, float *m_sh, float *v_sh,
    const int num_sh_coef, bool *d_mask, int *d_write_index, curandState *state) {
  if (!(use_delete || use_split || use_clone))
    return;

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane_id = idx % 32;

  unsigned active_mask = __ballot_sync(0xFFFFFFFF, idx < N);

  if (idx >= N)
    return;

  // Keep Gaussians that contribute to image
  const float op = opacity[idx];
  const int accum_count = grad_accum_count[idx];
  float uv_grad_avg_norm;
  if (accum_count > 0) {
    uv_grad_avg_norm = uv_grad_accum[idx] / accum_count;
  } else {
    uv_grad_avg_norm = 0.0f;
  }

  const float3 exp_scale = {expf(scale[idx * 3 + 0]), expf(scale[idx * 3 + 1]), expf(scale[idx * 3 + 2])};
  const float scale_max = fmaxf(exp_scale.x, fmaxf(exp_scale.y, exp_scale.z));

  // Delete
  bool keep = true;
  if (use_delete) {
    keep = keep_test(op, accum_count, scale_max, extent);
    d_mask[idx] = keep;
  }

  // Densify (Clone and Split)
  const float3 xyz_grad = {xyz_grad_accum[idx * 3 + 0], xyz_grad_accum[idx * 3 + 1], xyz_grad_accum[idx * 3 + 2]};
  float3 xyz_grad_avg;
  if (accum_count > 0) {
    xyz_grad_avg = {xyz_grad.x / accum_count, xyz_grad.y / accum_count, xyz_grad.z / accum_count};
  } else {
    xyz_grad_avg = {0.0f, 0.0f, 0.0f};
  }

  const bool densify = keep && (uv_grad_avg_norm > grad_threshold);

  // Clone
  const bool clone = densify && (scale_max <= percent_dense * extent);

  if (use_clone) {
    const int write_idx = get_strided_write_index(1, clone, lane_id, active_mask, d_write_index);
    if (clone && (write_idx < max_gaussians + 2)) {
      d_mask[write_idx] = true;
      // xyz
      const float new_x = xyz[idx * 3 + 0] - xyz_grad_avg.x * 0.01f;
      const float new_y = xyz[idx * 3 + 1] - xyz_grad_avg.y * 0.01f;
      const float new_z = xyz[idx * 3 + 2] - xyz_grad_avg.z * 0.01f;
      xyz[write_idx * 3 + 0] = new_x;
      xyz[write_idx * 3 + 1] = new_y;
      xyz[write_idx * 3 + 2] = new_z;
      // rgb
#pragma unroll
      for (int j = 0; j < 3; j++)
        rgb[write_idx * 3 + j] = rgb[idx * 3 + j];
      // opacity
      opacity[write_idx] = opacity[idx];
      // scale
#pragma unroll
      for (int j = 0; j < 3; j++)
        scale[write_idx * 3 + j] = scale[idx * 3 + j];
      // quaternion
#pragma unroll
      for (int j = 0; j < 4; j++)
        quaternion[write_idx * 4 + j] = quaternion[idx * 4 + j];
      // sh
      if (num_sh_coef > 0) {
        for (int j = 0; j < num_sh_coef * 3; j++)
          sh[write_idx * num_sh_coef * 3 + j] = sh[idx * num_sh_coef * 3 + j];
      }
      // optimizer moment vectors
#pragma unroll
      for (int j = 0; j < 3; j++) {
        m_xyz[write_idx * 3 + j] = 0.0f;
        v_xyz[write_idx * 3 + j] = 0.0f;
      }
#pragma unroll
      for (int j = 0; j < 3; j++) {
        m_rgb[write_idx * 3 + j] = 0.0f;
        v_rgb[write_idx * 3 + j] = 0.0f;
      }
      m_op[write_idx] = 0.0f;
      v_op[write_idx] = 0.0f;
#pragma unroll
      for (int j = 0; j < 3; j++) {
        m_scale[write_idx * 3 + j] = 0.0f;
        v_scale[write_idx * 3 + j] = 0.0f;
      }
#pragma unroll
      for (int j = 0; j < 4; j++) {
        m_quat[write_idx * 4 + j] = 0.0f;
        v_quat[write_idx * 4 + j] = 0.0f;
      }
      if (num_sh_coef > 0) {
        for (int j = 0; j < num_sh_coef * 3; j++) {
          m_sh[write_idx * num_sh_coef * 3 + j] = 0.0f;
          v_sh[write_idx * num_sh_coef * 3 + j] = 0.0f;
        }
      }
    }
  }

  // Split
  const bool split = densify && (scale_max > percent_dense * extent);

  if (use_split) {
    const int write_idx = get_strided_write_index(num_split_samples, split, lane_id, active_mask, d_write_index);
    if (split && (write_idx < max_gaussians + 2)) {
      // remove split Gaussian
      d_mask[idx] = false;

      // xyz
      const float new_x = xyz[idx * 3 + 0];
      const float new_y = xyz[idx * 3 + 1];
      const float new_z = xyz[idx * 3 + 2];

      // Quaternion to Rotation
      const int quat_base_idx = 4 * idx;
      float w = quaternion[quat_base_idx + 0];
      float x = quaternion[quat_base_idx + 1];
      float y = quaternion[quat_base_idx + 2];
      float z = quaternion[quat_base_idx + 3];

      const float inv_norm = rsqrtf(w * w + x * x + y * y + z * z);
      w *= inv_norm;
      x *= inv_norm;
      y *= inv_norm;
      z *= inv_norm;

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

      // Sample and split Gaussian
      for (int i = 0; i < num_split_samples; i++) {
        // add new Gaussians
        d_mask[write_idx + i] = true;
        const float v_x = curand_normal(&state[idx]) * exp_scale.x;
        const float v_y = curand_normal(&state[idx]) * exp_scale.y;
        const float v_z = curand_normal(&state[idx]) * exp_scale.z;

        const float rot_v_x = v_x * r00 + v_y * r01 + v_z * r02;
        const float rot_v_y = v_x * r10 + v_y * r11 + v_z * r12;
        const float rot_v_z = v_x * r20 + v_y * r21 + v_z * r22;

        // Write new Gaussian

        // xyz
        xyz[(write_idx + i) * 3 + 0] = new_x + rot_v_x;
        xyz[(write_idx + i) * 3 + 1] = new_y + rot_v_y;
        xyz[(write_idx + i) * 3 + 2] = new_z + rot_v_z;
        // rgb
#pragma unroll
        for (int j = 0; j < 3; j++)
          rgb[(write_idx + i) * 3 + j] = rgb[idx * 3 + j];
        // opacity
        opacity[write_idx + i] = opacity[idx];
        // scale
        scale[(write_idx + i) * 3 + 0] = logf(exp_scale.x / split_scale_factor);
        scale[(write_idx + i) * 3 + 1] = logf(exp_scale.y / split_scale_factor);
        scale[(write_idx + i) * 3 + 2] = logf(exp_scale.z / split_scale_factor);
        // quaternion
#pragma unroll
        for (int j = 0; j < 4; j++)
          quaternion[(write_idx + i) * 4 + j] = quaternion[idx * 4 + j];
        // sh
        if (num_sh_coef > 0) {
          for (int j = 0; j < num_sh_coef * 3; j++)
            sh[(write_idx + i) * num_sh_coef * 3 + j] = sh[idx * num_sh_coef * 3 + j];
        }

        // optimizer moment vectors
#pragma unroll
        for (int j = 0; j < 3; j++) {
          m_xyz[(write_idx + i) * 3 + j] = 0.0f;
          v_xyz[(write_idx + i) * 3 + j] = 0.0f;
        }
#pragma unroll
        for (int j = 0; j < 3; j++) {
          m_rgb[(write_idx + i) * 3 + j] = 0.0f;
          v_rgb[(write_idx + i) * 3 + j] = 0.0f;
        }
        m_op[(write_idx + i)] = 0.0f;
        v_op[(write_idx + i)] = 0.0f;
#pragma unroll
        for (int j = 0; j < 3; j++) {
          m_scale[(write_idx + i) * 3 + j] = 0.0f;
          v_scale[(write_idx + i) * 3 + j] = 0.0f;
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
          m_quat[(write_idx + i) * 4 + j] = 0.0f;
          v_quat[(write_idx + i) * 4 + j] = 0.0f;
        }
        if (num_sh_coef > 0) {
          for (int j = 0; j < num_sh_coef * 3; j++) {
            m_sh[(write_idx + i) * num_sh_coef * 3 + j] = 0.0f;
            v_sh[(write_idx + i) * num_sh_coef * 3 + j] = 0.0f;
          }
        }
      }
    }
  }
}

int adaptive_density(const int N, const int iter, const int num_sh_coef, const int max_gaussians, const bool use_delete,
                     const bool use_clone, const bool use_split, const float delete_opacity_threshold,
                     const float grad_threshold, const float scene_extent, const float percent_dense,
                     const int num_split_samples, const float split_scale_factor, const float *uv_grad_accum,
                     const int *grad_accum_count, float *scale, bool *d_mask, const float *xyz_grad_accum, float *xyz,
                     float *rgb, float *sh, float *opacity, float *quaternion, float *m_xyz, float *v_xyz, float *m_rgb,
                     float *v_rgb, float *m_op, float *v_op, float *m_scale, float *v_scale, float *m_quat,
                     float *v_quat, float *m_sh, float *v_sh, cudaStream_t stream) {
  const int threads_per_block = 256;

  // Calculate the number of blocks needed to cover all N Gaussians.
  const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

  const dim3 gridsize(num_blocks, 1, 1);
  const dim3 blocksize(threads_per_block, 1, 1);

  // Create random states
  curandState *curand_states;
  CHECK_CUDA(cudaMalloc(&curand_states, threads_per_block * num_blocks * sizeof(curandState)));

  setup_states<<<gridsize, blocksize>>>(curand_states, (uint64_t)time(NULL));

  // Reset mask before adaptive density checks
  CHECK_CUDA(cudaMemset(d_mask, false, max_gaussians * sizeof(bool)));

  // Set global index to end of array
  int *global_index;
  CHECK_CUDA(cudaMalloc(&global_index, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(global_index, &N, sizeof(int), cudaMemcpyHostToDevice));

  fused_adaptive_density_kernel<<<gridsize, blocksize, 0, stream>>>(
      N, max_gaussians, use_delete, use_clone, use_split, delete_opacity_threshold, scene_extent, percent_dense,
      grad_threshold, num_split_samples, split_scale_factor, uv_grad_accum, xyz_grad_accum, grad_accum_count, xyz, rgb,
      sh, opacity, scale, quaternion, m_xyz, v_xyz, m_rgb, v_rgb, m_op, v_op, m_scale, v_scale, m_quat, v_quat, m_sh,
      v_sh, num_sh_coef, d_mask, global_index, curand_states);

  // Free memory
  CHECK_CUDA(cudaFree(curand_states));

  int total_size = 0;
  CHECK_CUDA(cudaMemcpy(&total_size, global_index, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(global_index));

  return fmin(max_gaussians, total_size);
}
