// raster.cu

#include "gsplat_cuda/raster.cuh"

#include "gsplat_cuda/cuda_data.cuh"
#include "gsplat_cuda/cuda_forward.cuh"
#include <thrust/count.h>
#include <thrust/device_vector.h>

void rasterize_image(const int num_gaussians, const Camera &camera, const ConfigParameters &config,
                     CameraParameters &camera_parameters, GaussianParameters &gaussians, ForwardPassData &pass_data,
                     const float bg_color, const int l_max) {
  const int width = (int)camera.width;
  const int height = (int)camera.height;

  // Initialize mask
  pass_data.d_mask.resize(num_gaussians);

  // Step 1: Projections and Culling
  camera_extrinsic_projection(thrust::raw_pointer_cast(gaussians.d_xyz.data()),
                              thrust::raw_pointer_cast(camera_parameters.d_T.data()), num_gaussians,
                              thrust::raw_pointer_cast(pass_data.d_xyz_c.data()));
  camera_intrinsic_projection(thrust::raw_pointer_cast(pass_data.d_xyz_c.data()),
                              thrust::raw_pointer_cast(camera_parameters.d_K.data()), num_gaussians,
                              thrust::raw_pointer_cast(pass_data.d_uv.data()));
  cull_gaussians(thrust::raw_pointer_cast(pass_data.d_uv.data()), thrust::raw_pointer_cast(pass_data.d_xyz_c.data()),
                 num_gaussians, config.near_thresh, config.far_thresh, config.cull_mask_padding, width, height,
                 thrust::raw_pointer_cast(pass_data.d_mask.data()));

  pass_data.num_culled = thrust::count(pass_data.d_mask.begin(), pass_data.d_mask.end(), true);

  if (pass_data.num_culled == 0) {
    fprintf(stderr, "Error no Gaussians in view for image\n");
    exit(EXIT_FAILURE);
  }

  // Select SH coefficients from mask
  thrust::device_vector<float> d_sh_selected;
  switch (l_max) {
  case 0:
    break;
  case 1:
    d_sh_selected = compact_masked_array<9>(gaussians.d_sh, pass_data.d_mask, pass_data.num_culled);
  case 2:
    d_sh_selected = compact_masked_array<24>(gaussians.d_sh, pass_data.d_mask, pass_data.num_culled);
  case 3:
    d_sh_selected = compact_masked_array<45>(gaussians.d_sh, pass_data.d_mask, pass_data.num_culled);
  default:
    fprintf(stderr, "Error SH band is invalid\n");
    exit(EXIT_FAILURE);
  }

  // Step 2: Filter Gaussians based on the culling mask
  auto d_rgb_selected = compact_masked_array<3>(gaussians.d_rgb, pass_data.d_mask, pass_data.num_culled);
  auto d_uv_selected = compact_masked_array<2>(pass_data.d_uv, pass_data.d_mask, pass_data.num_culled);
  auto d_opacity_selected = compact_masked_array<1>(gaussians.d_opacity, pass_data.d_mask, pass_data.num_culled);
  auto d_xyz_c_selected = compact_masked_array<3>(pass_data.d_xyz_c, pass_data.d_mask, pass_data.num_culled);
  auto d_quaternion_selected = compact_masked_array<4>(gaussians.d_quaternion, pass_data.d_mask, pass_data.num_culled);
  auto d_scale_selected = compact_masked_array<3>(gaussians.d_scale, pass_data.d_mask, pass_data.num_culled);
  auto d_xyz_selected = compact_masked_array<3>(gaussians.d_xyz, pass_data.d_mask, pass_data.num_culled);

  // Step 3; Compute final RGB values from spherical harmonics
  pass_data.d_precomputed_rgb.resize(pass_data.num_culled * 3);

  precompute_spherical_harmonics(thrust::raw_pointer_cast(d_xyz_c_selected.data()),
                                 thrust::raw_pointer_cast(d_sh_selected.data()),
                                 thrust::raw_pointer_cast(d_rgb_selected.data()), l_max, pass_data.num_culled,
                                 thrust::raw_pointer_cast(pass_data.d_precomputed_rgb.data()));

  // Step 4: Compute Covariance and Conics
  pass_data.d_sigma.resize(pass_data.num_culled * 9);
  pass_data.d_conic.resize(pass_data.num_culled * 3);
  pass_data.d_J.resize(pass_data.num_culled * 6);

  compute_sigma(thrust::raw_pointer_cast(d_quaternion_selected.data()),
                thrust::raw_pointer_cast(d_scale_selected.data()), pass_data.num_culled,
                thrust::raw_pointer_cast(pass_data.d_sigma.data()));
  compute_conic(thrust::raw_pointer_cast(d_xyz_c_selected.data()),
                thrust::raw_pointer_cast(camera_parameters.d_K.data()),
                thrust::raw_pointer_cast(pass_data.d_sigma.data()),
                thrust::raw_pointer_cast(camera_parameters.d_T.data()), pass_data.num_culled,
                thrust::raw_pointer_cast(pass_data.d_J.data()), thrust::raw_pointer_cast(pass_data.d_conic.data()));

  // Step 5: Sort Gaussians by tile
  const int n_tiles_x = (width + TILE_SIZE_FWD - 1) / TILE_SIZE_FWD;
  const int n_tiles_y = (height + TILE_SIZE_FWD - 1) / TILE_SIZE_FWD;
  const int n_tiles = n_tiles_x * n_tiles_y;
  size_t sorted_gaussian_size = 0;
  get_sorted_gaussian_list(thrust::raw_pointer_cast(d_uv_selected.data()),
                           thrust::raw_pointer_cast(d_xyz_c_selected.data()),
                           thrust::raw_pointer_cast(pass_data.d_conic.data()), n_tiles_x, n_tiles_y, config.mh_dist,
                           pass_data.num_culled, sorted_gaussian_size, nullptr, nullptr);

  pass_data.d_splat_start_end_idx_by_tile_idx.resize(n_tiles + 1);
  pass_data.d_sorted_gaussians.resize(sorted_gaussian_size);

  get_sorted_gaussian_list(
      thrust::raw_pointer_cast(d_uv_selected.data()), thrust::raw_pointer_cast(d_xyz_c_selected.data()),
      thrust::raw_pointer_cast(pass_data.d_conic.data()), n_tiles_x, n_tiles_y, config.mh_dist, pass_data.num_culled,
      sorted_gaussian_size, thrust::raw_pointer_cast(pass_data.d_sorted_gaussians.data()),
      thrust::raw_pointer_cast(pass_data.d_splat_start_end_idx_by_tile_idx.data()));

  // Step 6: Render the final image
  pass_data.d_image_buffer.resize(height * width * 3);
  pass_data.d_weight_per_pixel.resize(height * width);
  pass_data.d_splats_per_pixel.resize(height * width);

  render_image(thrust::raw_pointer_cast(d_uv_selected.data()), thrust::raw_pointer_cast(d_opacity_selected.data()),
               thrust::raw_pointer_cast(pass_data.d_conic.data()),
               thrust::raw_pointer_cast(pass_data.d_precomputed_rgb.data()), bg_color,
               thrust::raw_pointer_cast(pass_data.d_sorted_gaussians.data()),
               thrust::raw_pointer_cast(pass_data.d_splat_start_end_idx_by_tile_idx.data()), width, height,
               thrust::raw_pointer_cast(pass_data.d_splats_per_pixel.data()),
               thrust::raw_pointer_cast(pass_data.d_weight_per_pixel.data()),
               thrust::raw_pointer_cast(pass_data.d_image_buffer.data()));
}
