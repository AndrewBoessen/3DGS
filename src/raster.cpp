#include "gsplat/raster.hpp"
#include "gsplat/cuda_forward.hpp"
#include <vector>

void rasterize_image(ConfigParameters config, Gaussians gaussians, Image image, Camera camera, float *out_image) {
  // Use a fixed number of streams for parallel processing
  constexpr int NUM_STREAMS = 4;

  int N = gaussians.size();
  if (N == 0) {
    return; // No gaussians to render
  }

  // Gaussian parameters on the device
  float *d_xyz, *d_rgb, *d_opacity, *d_scale, *d_quaternion;
  CHECK_CUDA(cudaMalloc(&d_xyz, N * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_rgb, N * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_opacity, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_scale, N * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_quaternion, N * 4 * sizeof(float)));

  // Camera parameters on the device
  float *d_K;
  const int width = (int)camera.width;
  const int height = (int)camera.height;
  CHECK_CUDA(cudaMalloc(&d_K, 9 * sizeof(float)));

  // Extrinsic camera matrix T = [R|t]
  float *d_T;
  CHECK_CUDA(cudaMalloc(&d_T, 12 * sizeof(float)));

  // Copy Gaussian data from host to device
  CHECK_CUDA(cudaMemcpy(d_xyz, gaussians.xyz.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_rgb, gaussians.rgb.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_opacity, gaussians.opacity.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_scale, gaussians.scale.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_quaternion, gaussians.quaternion.data(), N * 4 * sizeof(float), cudaMemcpyHostToDevice));

  // Prepare and copy camera intrinsics (K) to device
  float h_K[9] = {(float)camera.params[0],
                  0.f,
                  (float)camera.params[2],
                  0.f,
                  (float)camera.params[1],
                  (float)camera.params[3],
                  0.f,
                  0.f,
                  1.f};
  CHECK_CUDA(cudaMemcpy(d_K, h_K, 9 * sizeof(float), cudaMemcpyHostToDevice));

  // Prepare and copy camera extrinsics (T) to device
  Eigen::Matrix3d rot_mat_d = image.QvecToRotMat();
  Eigen::Vector3d t_vec_d = image.tvec;
  float h_T[12];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      h_T[i * 4 + j] = (float)rot_mat_d(i, j);
    }
    h_T[i * 4 + 3] = (float)t_vec_d(i);
  }
  CHECK_CUDA(cudaMemcpy(d_T, h_T, 12 * sizeof(float), cudaMemcpyHostToDevice));

  // Temporary buffers for processing
  float *d_uv, *d_xyz_c;
  bool *d_mask;
  CHECK_CUDA(cudaMalloc(&d_uv, N * 2 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_xyz_c, N * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_mask, N * sizeof(bool)));

  // Create CUDA streams for parallel execution
  std::vector<cudaStream_t> streams(NUM_STREAMS);
  for (int i = 0; i < NUM_STREAMS; ++i) {
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
  }

  // Step 1: Projections and Culling (Parallelized across streams)
  int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;
  for (int i = 0; i < NUM_STREAMS; ++i) {
    int offset = i * chunk_size;
    int size = std::min(chunk_size, N - offset);
    if (size <= 0)
      continue;

    cudaStream_t stream = streams[i];
    camera_extrinsic_projection(d_xyz + offset * 3, d_T, size, d_xyz_c + offset * 3, stream);
    camera_intrinsic_projection(d_xyz_c + offset * 3, d_K, size, d_uv + offset * 2, stream);
    cull_gaussians(d_uv + offset * 2, d_xyz_c + offset * 3, size, config.near_thresh, config.far_thresh,
                   config.cull_mask_padding, width, height, d_mask + offset, stream);
  }

  CHECK_CUDA(cudaDeviceSynchronize()); // Sync streams to ensure mask is complete

  // Step 2: Filter Gaussians based on the culling mask using CUB
  int *d_num_culled;
  CHECK_CUDA(cudaMalloc(&d_num_culled, sizeof(int)));

  float *d_xyz_culled, *d_rgb_culled, *d_opacity_culled, *d_scale_culled, *d_quaternion_culled, *d_uv_culled,
      *d_xyz_c_culled;
  CHECK_CUDA(cudaMalloc(&d_xyz_culled, N * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_rgb_culled, N * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_opacity_culled, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_scale_culled, N * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_quaternion_culled, N * 4 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_uv_culled, N * 2 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_xyz_c_culled, N * 3 * sizeof(float)));

  filter_gaussians_by_mask(N, d_mask, d_xyz, d_rgb, d_opacity, d_scale, d_quaternion, d_uv, d_xyz_c, d_xyz_culled,
                           d_rgb_culled, d_opacity_culled, d_scale_culled, d_quaternion_culled, d_uv_culled,
                           d_xyz_c_culled, d_num_culled);

  int N_culled;
  CHECK_CUDA(cudaMemcpy(&N_culled, d_num_culled, sizeof(int), cudaMemcpyDeviceToHost));

  if (N_culled == 0) {
    // Cleanup and return if no gaussians are left
    for (int i = 0; i < NUM_STREAMS; ++i)
      CHECK_CUDA(cudaStreamDestroy(streams[i]));
    // Free all memory
    CHECK_CUDA(cudaFree(d_xyz));
    CHECK_CUDA(cudaFree(d_rgb));
    CHECK_CUDA(cudaFree(d_opacity));
    CHECK_CUDA(cudaFree(d_scale));
    CHECK_CUDA(cudaFree(d_quaternion));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_T));
    CHECK_CUDA(cudaFree(d_uv));
    CHECK_CUDA(cudaFree(d_xyz_c));
    CHECK_CUDA(cudaFree(d_mask));
    CHECK_CUDA(cudaFree(d_num_culled));
    CHECK_CUDA(cudaFree(d_xyz_culled));
    CHECK_CUDA(cudaFree(d_rgb_culled));
    CHECK_CUDA(cudaFree(d_opacity_culled));
    CHECK_CUDA(cudaFree(d_scale_culled));
    CHECK_CUDA(cudaFree(d_quaternion_culled));
    CHECK_CUDA(cudaFree(d_uv_culled));
    CHECK_CUDA(cudaFree(d_xyz_c_culled));

    return;
  }

  // Step 3: Compute Covariance and Conics (Parallelized across streams)
  float *d_sigma, *d_conic, *d_J;
  CHECK_CUDA(cudaMalloc(&d_sigma, N_culled * 6 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_conic, N_culled * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_J, N_culled * 9 * sizeof(float)));

  chunk_size = (N_culled + NUM_STREAMS - 1) / NUM_STREAMS;
  for (int i = 0; i < NUM_STREAMS; ++i) {
    int offset = i * chunk_size;
    int size = std::min(chunk_size, N_culled - offset);
    if (size <= 0)
      continue;

    cudaStream_t stream = streams[i];
    compute_sigma(d_quaternion_culled + offset * 4, d_scale_culled + offset * 3, size, d_sigma + offset * 6, stream);
    compute_conic(d_xyz_c_culled + offset * 3, d_K, d_sigma + offset * 6, d_T, size, d_J + offset * 9,
                  d_conic + offset * 3, stream);
  }

  CHECK_CUDA(cudaDeviceSynchronize()); // Sync streams for sorting

  // Step 4: Sort Gaussians by tile
  const int n_tiles_x = (width + 15) / 16;
  const int n_tiles_y = (height + 15) / 16;
  const int n_tiles = n_tiles_x * n_tiles_y;

  int *d_sorted_gaussians, *d_splat_start_end_idx_by_tile_idx;
  size_t sorted_gaussian_bytes = 0;

  CHECK_CUDA(cudaMalloc(&d_splat_start_end_idx_by_tile_idx, n_tiles * 2 * sizeof(int)));

  get_sorted_gaussian_list(d_uv_culled, d_xyz_c_culled, d_conic, n_tiles_x, n_tiles_y, config.mh_dist, N_culled,
                           sorted_gaussian_bytes, nullptr, d_splat_start_end_idx_by_tile_idx);

  CHECK_CUDA(cudaMalloc(&d_sorted_gaussians, sorted_gaussian_bytes));

  get_sorted_gaussian_list(d_uv_culled, d_xyz_c_culled, d_conic, n_tiles_x, n_tiles_y, config.mh_dist, N_culled,
                           sorted_gaussian_bytes, d_sorted_gaussians, d_splat_start_end_idx_by_tile_idx);

  // Step 5: Render the final image
  float *d_image_buffer, *d_weight_per_pixel;
  CHECK_CUDA(cudaMalloc(&d_image_buffer, width * height * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_weight_per_pixel, width * height * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_image_buffer, 0, width * height * 3 * sizeof(float)));

  render_image(d_uv_culled, d_opacity_culled, d_conic, d_rgb_culled, 1.0f, d_sorted_gaussians,
               d_splat_start_end_idx_by_tile_idx, width, height, d_weight_per_pixel, d_image_buffer);

  // Copy final image from device to host
  CHECK_CUDA(cudaMemcpy(out_image, d_image_buffer, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  // Cleanup: Free all CUDA memory and destroy streams
  CHECK_CUDA(cudaFree(d_xyz));
  CHECK_CUDA(cudaFree(d_rgb));
  CHECK_CUDA(cudaFree(d_opacity));
  CHECK_CUDA(cudaFree(d_scale));
  CHECK_CUDA(cudaFree(d_quaternion));
  CHECK_CUDA(cudaFree(d_K));
  CHECK_CUDA(cudaFree(d_T));
  CHECK_CUDA(cudaFree(d_uv));
  CHECK_CUDA(cudaFree(d_xyz_c));
  CHECK_CUDA(cudaFree(d_mask));
  CHECK_CUDA(cudaFree(d_num_culled));
  CHECK_CUDA(cudaFree(d_xyz_culled));
  CHECK_CUDA(cudaFree(d_rgb_culled));
  CHECK_CUDA(cudaFree(d_opacity_culled));
  CHECK_CUDA(cudaFree(d_scale_culled));
  CHECK_CUDA(cudaFree(d_quaternion_culled));
  CHECK_CUDA(cudaFree(d_uv_culled));
  CHECK_CUDA(cudaFree(d_xyz_c_culled));
  CHECK_CUDA(cudaFree(d_sigma));
  CHECK_CUDA(cudaFree(d_conic));
  CHECK_CUDA(cudaFree(d_J));
  CHECK_CUDA(cudaFree(d_sorted_gaussians));
  CHECK_CUDA(cudaFree(d_splat_start_end_idx_by_tile_idx));
  CHECK_CUDA(cudaFree(d_image_buffer));
  CHECK_CUDA(cudaFree(d_weight_per_pixel));

  for (int i = 0; i < NUM_STREAMS; ++i) {
    CHECK_CUDA(cudaStreamDestroy(streams[i]));
  }
}
