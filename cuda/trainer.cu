// trainer.cu

#include "gsplat_cuda/adaptive_density.cuh"
#include "gsplat_cuda/cuda_backward.cuh"
#include "gsplat_cuda/cuda_data.cuh"
#include "gsplat_cuda/cuda_forward.cuh"
#include "gsplat_cuda/optimizer.cuh"
#include "gsplat_cuda/raster.cuh"
#include "gsplat_cuda/trainer.cuh"
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

void save_image(const std::string &filename, const thrust::device_vector<float> &d_image_buffer, int width,
                int height) {
  const size_t expected_size = static_cast<size_t>(width * height * 3);

  if (d_image_buffer.size() != expected_size) {
    throw std::runtime_error("Device image buffer size does not match expected size (width*height*3).");
  }

  thrust::host_vector<float> h_image_data(expected_size);

  try {
    thrust::copy(d_image_buffer.begin(), d_image_buffer.end(), h_image_data.begin());
  } catch (const std::exception &e) {
    // Catch potential errors from the copy operation.
    throw std::runtime_error("Failed to copy image data from device to host: " + std::string(e.what()));
  }

  cv::Mat float_image(height, width, CV_32FC3, h_image_data.data());

  cv::Mat bgr_image;
  float_image.convertTo(bgr_image, CV_8UC3, 255.0);
  cv::cvtColor(bgr_image, bgr_image, cv::COLOR_RGB2BGR);

  if (cv::imwrite(filename, bgr_image)) {
    std::cout << "Saved image to " << filename << std::endl;
  } else {
    std::cerr << "Error: Failed to save image to " << filename << std::endl;
  }
}

void Trainer::test_train_split() {
  const int split = config.test_split_ratio;

  // Ensure the destination vectors are empty before splitting
  test_images.clear();
  train_images.clear();

  if (images.empty()) {
    return;
  }

  // Convert the map of images to a vector to allow for sorting
  std::vector<Image> all_images;
  all_images.reserve(images.size());
  for (const auto &pair : images) {
    all_images.push_back(pair.second);
  }

  // Sort the images by name to ensure a deterministic split across runs
  std::sort(all_images.begin(), all_images.end(), [](const Image &a, const Image &b) { return a.name < b.name; });

  // Handle edge cases for the split ratio
  if (split <= 0) {
    train_images = all_images;
  } else {
    // Use the "every N-th image" strategy for a representative split
    for (size_t i = 0; i < all_images.size(); ++i) {
      if (i % split == 0) {
        test_images.push_back(all_images[i]);
      } else {
        train_images.push_back(all_images[i]);
      }
    }
  }
}

void Trainer::reset_grad_accum(GradientAccumulators &accumulators) {
  thrust::fill_n(accumulators.d_uv_grad_accum.begin(), num_gaussians, 0.0f);
  thrust::fill_n(accumulators.d_xyz_grad_accum.begin(), num_gaussians, 0.0f);
  thrust::fill_n(accumulators.d_grad_accum_dur.begin(), num_gaussians, 0);
}

void Trainer::reset_opacity(GaussianParameters &gaussians) {
  // set all Gaussian opacity
  const double opc = config.reset_opacity_value;
  const float new_opc = log(opc) - log(1.0f - opc);

  thrust::fill_n(gaussians.d_opacity.begin(), num_gaussians, new_opc);
}

struct SHIndexMapper {
  const int old_coeffs_per_gaussian;
  const int new_coeffs_per_gaussian;

  SHIndexMapper(int old_coeffs, int new_coeffs)
      : old_coeffs_per_gaussian(old_coeffs), new_coeffs_per_gaussian(new_coeffs) {}

  __host__ __device__ int operator()(int src_idx) const {
    int gaussian_idx = src_idx / old_coeffs_per_gaussian;
    int coeff_idx_in_gaussian = src_idx % old_coeffs_per_gaussian;
    return gaussian_idx * new_coeffs_per_gaussian + coeff_idx_in_gaussian;
  }
};

void Trainer::add_sh_band(GaussianParameters &gaussians) {
  if (l_max >= config.max_sh_band)
    return;

  if (l_max == 0) {
    thrust::fill(gaussians.d_sh.begin(), gaussians.d_sh.end(), 0.0f);
    l_max++;
    return;
  }
  const int curr_param_count = (l_max + 1) * (l_max + 1) - 1;
  const int new_param_count = (l_max + 2) * (l_max + 2) - 1;

  const int old_coeffs_per_gaussian = curr_param_count * 3;
  const int new_coeffs_per_gaussian = new_param_count * 3;

  try {
    thrust::device_vector<float> temp_sh(num_gaussians * old_coeffs_per_gaussian);
    thrust::copy_n(gaussians.d_sh.begin(), temp_sh.size(), temp_sh.begin());

    thrust::device_vector<int> destination_map(temp_sh.size());

    // 1. Instantiate the functor with the captured state
    SHIndexMapper mapper(old_coeffs_per_gaussian, new_coeffs_per_gaussian);

    // 2. Pass the functor instance to thrust::transform
    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(temp_sh.size()),
                      destination_map.begin(), mapper);

    thrust::fill(gaussians.d_sh.begin(), gaussians.d_sh.end(), 0.0f);

    thrust::scatter(temp_sh.begin(), temp_sh.end(), destination_map.begin(), gaussians.d_sh.begin());

  } catch (const std::exception &e) {
    fprintf(stderr, "Error during SH band expansion: %s\n", e.what());
    exit(EXIT_FAILURE);
  }

  l_max++;
}

void Trainer::adaptive_density_step(GaussianParameters &gaussians, GradientAccumulators &accumulators) {}

float Trainer::backward_pass(const Image &curr_image, const Camera &curr_camera, GaussianGradients &gradients,
                             ForwardPassData &pass_data, GaussianParameters &gaussians, CameraParameters &camera,
                             const float bg_color) {
  const int width = (int)curr_camera.width;
  const int height = (int)curr_camera.height;

  // Load ground truth image from file
  cv::Mat bgr_gt_image = cv::imread(curr_image.name, cv::IMREAD_COLOR);

  if (bgr_gt_image.empty()) {
    // Throw an exception if the image failed to load, preventing a halt.
    throw std::runtime_error("Failed to load image: " + curr_image.name);
  }
  if (bgr_gt_image.cols != width || bgr_gt_image.rows != height) {
    // Throw an exception if the image dimensions do not match expectations.
    throw std::runtime_error("Image dimensions mismatch for " + curr_image.name + ". Expected " +
                             std::to_string(width) + "x" + std::to_string(height) + ", but got " +
                             std::to_string(bgr_gt_image.cols) + "x" + std::to_string(bgr_gt_image.rows));
  }

  // Convert image to the required format (RGB, 32-bit float)
  cv::Mat rgb_gt_image;
  cv::cvtColor(bgr_gt_image, rgb_gt_image, cv::COLOR_BGR2RGB);
  cv::Mat float_gt_image;
  rgb_gt_image.convertTo(float_gt_image, CV_32FC3, 1.0 / 255.0);

  thrust::device_vector<float> d_gt_image(height * width * 3);
  thrust::copy(float_gt_image.data, float_gt_image.data + height * width * 3, d_gt_image.begin());

  // Compute loss and image gradient
  thrust::device_vector<float> d_grad_image(height * width * 3);
  float loss =
      fused_loss(thrust::raw_pointer_cast(pass_data.d_image_buffer.data()), thrust::raw_pointer_cast(d_gt_image.data()),
                 height, width, 3, config.ssim_frac, thrust::raw_pointer_cast(d_grad_image.data()));

  // Prepare data for gradients
  auto d_uv_selected = compact_masked_array<2>(pass_data.d_uv, pass_data.d_mask, pass_data.num_culled);
  auto d_opacity_selected = compact_masked_array<1>(gaussians.d_opacity, pass_data.d_mask, pass_data.num_culled);
  auto d_xyz_c_selected = compact_masked_array<3>(pass_data.d_xyz_c, pass_data.d_mask, pass_data.num_culled);
  auto d_quaternion_selected = compact_masked_array<4>(gaussians.d_quaternion, pass_data.d_mask, pass_data.num_culled);
  auto d_scale_selected = compact_masked_array<3>(gaussians.d_scale, pass_data.d_mask, pass_data.num_culled);
  auto d_xyz_selected = compact_masked_array<3>(gaussians.d_xyz, pass_data.d_mask, pass_data.num_culled);

  // Backpropagate gradients from image to Gaussian parameters
  render_image_backward(
      thrust::raw_pointer_cast(d_uv_selected.data()), thrust::raw_pointer_cast(d_opacity_selected.data()),
      thrust::raw_pointer_cast(pass_data.d_conic.data()), thrust::raw_pointer_cast(pass_data.d_precomputed_rgb.data()),
      bg_color, thrust::raw_pointer_cast(pass_data.d_sorted_gaussians.data()),
      thrust::raw_pointer_cast(pass_data.d_splat_start_end_idx_by_tile_idx.data()),
      thrust::raw_pointer_cast(pass_data.d_splats_per_pixel.data()),
      thrust::raw_pointer_cast(pass_data.d_weight_per_pixel.data()), thrust::raw_pointer_cast(d_grad_image.data()),
      width, height, thrust::raw_pointer_cast(gradients.d_grad_precompute_rgb.data()),
      thrust::raw_pointer_cast(gradients.d_grad_opacity.data()), thrust::raw_pointer_cast(gradients.d_grad_uv.data()),
      thrust::raw_pointer_cast(gradients.d_grad_conic.data()));

  precompute_spherical_harmonics_backward(thrust::raw_pointer_cast(d_xyz_c_selected.data()),
                                          thrust::raw_pointer_cast(gradients.d_grad_precompute_rgb.data()), l_max,
                                          pass_data.num_culled, thrust::raw_pointer_cast(gradients.d_grad_sh.data()),
                                          thrust::raw_pointer_cast(gradients.d_grad_rgb.data()));
  compute_conic_backward(
      thrust::raw_pointer_cast(pass_data.d_J.data()), thrust::raw_pointer_cast(pass_data.d_sigma.data()),
      thrust::raw_pointer_cast(camera.d_T.data()), thrust::raw_pointer_cast(gradients.d_grad_conic.data()),
      pass_data.num_culled, thrust::raw_pointer_cast(gradients.d_grad_J.data()),
      thrust::raw_pointer_cast(gradients.d_grad_sigma.data()));
  compute_projection_jacobian_backward(thrust::raw_pointer_cast(d_xyz_c_selected.data()),
                                       thrust::raw_pointer_cast(camera.d_K.data()),
                                       thrust::raw_pointer_cast(pass_data.d_J.data()), pass_data.num_culled,
                                       thrust::raw_pointer_cast(gradients.d_grad_xyz_c.data()));
  compute_sigma_backward(thrust::raw_pointer_cast(d_quaternion_selected.data()),
                         thrust::raw_pointer_cast(d_scale_selected.data()),
                         thrust::raw_pointer_cast(gradients.d_grad_sigma.data()), pass_data.num_culled,
                         thrust::raw_pointer_cast(gradients.d_grad_quaternion.data()),
                         thrust::raw_pointer_cast(gradients.d_grad_scale.data()));
  camera_intrinsic_projection_backward(thrust::raw_pointer_cast(d_xyz_c_selected.data()),
                                       thrust::raw_pointer_cast(camera.d_K.data()),
                                       thrust::raw_pointer_cast(gradients.d_grad_uv.data()), pass_data.num_culled,
                                       thrust::raw_pointer_cast(gradients.d_grad_xyz_c.data()));
  camera_extrinsic_projection_backward(thrust::raw_pointer_cast(d_xyz_selected.data()),
                                       thrust::raw_pointer_cast(camera.d_T.data()),
                                       thrust::raw_pointer_cast(gradients.d_grad_xyz_c.data()), pass_data.num_culled,
                                       thrust::raw_pointer_cast(gradients.d_grad_xyz.data()));

  return loss;
}

void optimizer_step(OptimizerParameters &optimizer, GaussianParameters &parameters, GaussianGradients &gradients,
                    GradientAccumulators &accumulators, const Camera &curr_camera) {
  // Select moment vectors by filtering based on the culled mask

  // Update parameters using the Adam optimizer step

  // Scatter the updated moment vectors back to the full tensors

  // Scatter the updated parameters back to the full tensors

  // Update Spherical Harmonics (SH) parameters if they are being used

  // Update gradient accumulators
}

void Trainer::train() {
  // Setup: Initialize CUDA data manager and data splits
  CudaDataManager cuda(config.max_gaussians);

  reset_grad_accum(cuda.accumulators);

  // Copy Gaussian data from host to device
  try {
    // Note the path to the device vectors is now cuda.gaussians.d_...
    const float *h_xyz = reinterpret_cast<float *>(gaussians.xyz.data());
    const float *h_rgb = reinterpret_cast<float *>(gaussians.rgb.data());
    const float *h_op = reinterpret_cast<float *>(gaussians.opacity.data());
    const float *h_scale = reinterpret_cast<float *>(gaussians.scale.data());
    const float *h_quat = reinterpret_cast<float *>(gaussians.quaternion.data());
    thrust::copy(h_xyz, h_xyz + num_gaussians * 3, cuda.gaussians.d_xyz.begin());
    thrust::copy(h_rgb, h_rgb + num_gaussians * 3, cuda.gaussians.d_rgb.begin());
    thrust::copy(h_op, h_op + num_gaussians, cuda.gaussians.d_opacity.begin());
    thrust::copy(h_scale, h_scale + num_gaussians * 3, cuda.gaussians.d_scale.begin());
    thrust::copy(h_quat, h_quat + num_gaussians * 4, cuda.gaussians.d_quaternion.begin());

  } catch (const std::exception &e) {
    fprintf(stderr, "Error copying data to device: %s\n", e.what());
    exit(EXIT_FAILURE);
  }

  std::random_device rd;                                             // obtain a random number from hardware
  std::mt19937 gen(rd());                                            // seed the generator
  std::uniform_int_distribution<> distr(0, train_images.size() - 1); // define the range

  // TRAINING LOOP
  while (iter < config.num_iters) {
    std::cout << "ITER " << iter << std::endl;
    std::cout << "NUM GAUSSIANS " << num_gaussians << std::endl;
    ForwardPassData pass_data;

    // Get current training image and camera
    Image curr_image = train_images[distr(gen)];
    Camera curr_camera = cameras[curr_image.camera_id];

    // Prepare and copy camera parameters to device
    float h_K[9] = {(float)curr_camera.params[0],
                    0.f,
                    (float)curr_camera.params[2],
                    0.f,
                    (float)curr_camera.params[1],
                    (float)curr_camera.params[3],
                    0.f,
                    0.f,
                    1.f};
    Eigen::Matrix3d rot_mat_d = curr_image.QvecToRotMat();
    Eigen::Vector3d t_vec_d = curr_image.tvec;
    float h_T[12];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j)
        h_T[i * 4 + j] = (float)rot_mat_d(i, j);
      h_T[i * 4 + 3] = (float)t_vec_d(i);
    }
    try {
      // Copy the intrinsics matrix (K)
      thrust::copy(h_K, h_K + 9, cuda.camera.d_K.begin());

      // Copy the extrinsics matrix (T)
      thrust::copy(h_T, h_T + 12, cuda.camera.d_T.begin());

    } catch (const std::exception &e) {
      fprintf(stderr, "Error copying camera data to device: %s\n", e.what());
      exit(EXIT_FAILURE);
    }

    // Background color
    float bg_color = 0.0f;
    if (config.use_background)
      bg_color = (iter % 255) / 255.0f;

    // Add SH bands
    if (iter % config.add_sh_band_interval == 0 && iter >= config.add_sh_band_interval)
      add_sh_band(cuda.gaussians);

    // --- FORWARD PASS via RASTERIZE MODULE ---
    rasterize_image(num_gaussians, curr_camera, config, cuda.camera, cuda.gaussians, pass_data, bg_color, l_max);

    if (pass_data.num_culled == 0) {
      std::cerr << "WARNING Image " << curr_image.id << " has no Gaussians in view" << std::endl;
      continue;
    }

    // --- BACKWARD PASS ---
    float loss =
        backward_pass(curr_image, curr_camera, cuda.gradients, pass_data, cuda.gaussians, cuda.camera, bg_color);
    std::cout << "LOSS TOTAL " << loss << std::endl;

    // --- OPTIMIZER STEP ---
    optimizer_step(cuda.optimizer, cuda.gaussians, cuda.gradients, cuda.accumulators, curr_camera);

    // --- ADAPTIVE DENSITY ---
    if (iter > config.adaptive_control_start && iter % config.adaptive_control_interval == 0 &&
        iter < config.adaptive_control_end) {
      adaptive_density_step(cuda.gaussians, cuda.accumulators);
      reset_grad_accum(cuda.accumulators);
    }

    if (iter > config.reset_opacity_start && iter % config.reset_opacity_interval == 0 &&
        iter < config.reset_opacity_end) {
      reset_opacity(cuda.gaussians);
      reset_grad_accum(cuda.accumulators);
    }

    // Increment iteration
    iter++;
  }
}
