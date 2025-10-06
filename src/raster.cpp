#include "gsplat/raster.hpp"
#include "gsplat/cuda_data.hpp"
#include "gsplat/cuda_forward.hpp"

void rasterize_image(const int num_gaussians, const Camera &camera, const ConfigParameters &config,
                     CudaDataManager &cuda, ForwardPassData &pass_data, const std::vector<cudaStream_t> &streams) {
  const int NUM_STREAMS = streams.size();
  const int width = (int)camera.width;
  const int height = (int)camera.height;

  // Step 1: Projections and Culling
  int offset = 0;
  for (int i = 0; i < NUM_STREAMS; ++i) {
    int remainder = num_gaussians % NUM_STREAMS;
    int size = num_gaussians / NUM_STREAMS + (i < remainder ? 1 : 0);
    if (size <= 0)
      continue;

    cudaStream_t stream = streams[i];
    camera_extrinsic_projection(cuda.d_xyz + offset * 3, cuda.d_T, size, cuda.d_xyz_c + offset * 3, stream);
    camera_intrinsic_projection(cuda.d_xyz_c + offset * 3, cuda.d_K, size, cuda.d_uv + offset * 2, stream);
    cull_gaussians(cuda.d_uv + offset * 2, cuda.d_xyz_c + offset * 3, size, config.near_thresh, config.far_thresh,
                   config.cull_mask_padding, width, height, cuda.d_mask + offset, stream);
    offset += size;
  }
  CHECK_CUDA(cudaDeviceSynchronize()); // Sync streams to ensure mask is complete

  // subtract 1 to account for band 0 being rgb
  const int num_sh_coef = (pass_data.l_max + 1) * (pass_data.l_max + 1) - 1;
  // Step 2: Filter Gaussians based on the culling mask
  filter_gaussians_by_mask(num_gaussians, num_sh_coef, cuda.d_mask, cuda.d_xyz, cuda.d_rgb, cuda.d_sh, cuda.d_opacity,
                           cuda.d_scale, cuda.d_quaternion, cuda.d_uv, cuda.d_xyz_c, cuda.d_xyz_culled,
                           cuda.d_rgb_culled, cuda.d_sh_culled, cuda.d_opacity_culled, cuda.d_scale_culled,
                           cuda.d_quaternion_culled, cuda.d_uv_culled, cuda.d_xyz_c_culled, &pass_data.num_culled);

  if (pass_data.num_culled == 0) {
    return; // No Gaussians in view
  }

  // Step 3; Compute final RGB values from spherical harmonics
  CHECK_CUDA(cudaMalloc(&pass_data.d_precomputed_rgb, pass_data.num_culled * 3 * sizeof(float)));

  offset = 0;
  for (int i = 0; i < NUM_STREAMS; ++i) {
    int remainder = pass_data.num_culled % NUM_STREAMS;
    int size = pass_data.num_culled / NUM_STREAMS + (i < remainder ? 1 : 0);
    if (size <= 0)
      continue;

    cudaStream_t stream = streams[i];
    precompute_spherical_harmonics(cuda.d_xyz_c_culled, cuda.d_sh_culled, cuda.d_rgb_culled, pass_data.l_max, size,
                                   pass_data.d_precomputed_rgb + offset * 3, stream);
    offset += size;
  }

  // Step 4: Compute Covariance and Conics
  CHECK_CUDA(cudaMalloc(&pass_data.d_sigma, pass_data.num_culled * 9 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&pass_data.d_conic, pass_data.num_culled * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&pass_data.d_J, pass_data.num_culled * 6 * sizeof(float)));

  offset = 0;
  for (int i = 0; i < NUM_STREAMS; ++i) {
    int remainder = pass_data.num_culled % NUM_STREAMS;
    int size = pass_data.num_culled / NUM_STREAMS + (i < remainder ? 1 : 0);
    if (size <= 0)
      continue;

    cudaStream_t stream = streams[i];
    compute_sigma(cuda.d_quaternion_culled + offset * 4, cuda.d_scale_culled + offset * 3, size,
                  pass_data.d_sigma + offset * 9, stream);
    compute_conic(cuda.d_xyz_c_culled + offset * 3, cuda.d_K, pass_data.d_sigma + offset * 9, cuda.d_T, size,
                  pass_data.d_J + offset * 6, pass_data.d_conic + offset * 3, stream);
    offset += size;
  }
  CHECK_CUDA(cudaDeviceSynchronize()); // Sync streams for sorting

  // Step 5: Sort Gaussians by tile
  const int n_tiles_x = (width + 15) / 16;
  const int n_tiles_y = (height + 15) / 16;
  const int n_tiles = n_tiles_x * n_tiles_y;
  size_t sorted_gaussian_bytes = 0;
  get_sorted_gaussian_list(cuda.d_uv_culled, cuda.d_xyz_c_culled, pass_data.d_conic, n_tiles_x, n_tiles_y,
                           config.mh_dist, pass_data.num_culled, sorted_gaussian_bytes, nullptr, nullptr);

  CHECK_CUDA(cudaMalloc(&pass_data.d_splat_start_end_idx_by_tile_idx, (n_tiles + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&pass_data.d_sorted_gaussians, sorted_gaussian_bytes));

  get_sorted_gaussian_list(cuda.d_uv_culled, cuda.d_xyz_c_culled, pass_data.d_conic, n_tiles_x, n_tiles_y,
                           config.mh_dist, pass_data.num_culled, sorted_gaussian_bytes, pass_data.d_sorted_gaussians,
                           pass_data.d_splat_start_end_idx_by_tile_idx);

  // Step 6: Render the final image
  CHECK_CUDA(cudaMalloc(&pass_data.d_image_buffer, width * height * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&pass_data.d_weight_per_pixel, width * height * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&pass_data.d_splats_per_pixel, width * height * sizeof(int)));
  CHECK_CUDA(cudaMemset(pass_data.d_image_buffer, 0.0f, width * height * 3 * sizeof(float)));

  render_image(cuda.d_uv_culled, cuda.d_opacity_culled, pass_data.d_conic, pass_data.d_precomputed_rgb, 1.0f,
               pass_data.d_sorted_gaussians, pass_data.d_splat_start_end_idx_by_tile_idx, width, height,
               pass_data.d_splats_per_pixel, pass_data.d_weight_per_pixel, pass_data.d_image_buffer);
}
