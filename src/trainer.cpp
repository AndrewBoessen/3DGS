// trainer.cpp

#include "gsplat/trainer.hpp"
#include "gsplat/cuda_backward.hpp"
#include "gsplat/cuda_data.hpp"
#include "gsplat/cuda_forward.hpp"
#include "gsplat/optimizer.hpp"
#include "gsplat/raster.hpp"
#include <format>
#include <iostream>

// Helper function to save an image from a device buffer
void save_image(const std::string &filename, const float *d_image_buffer, int width, int height) {
  // Create a host vector to hold the image data
  std::vector<float> h_image_data(width * height * 3);

  // Copy the image data from the GPU device to the host
  CHECK_CUDA(
      cudaMemcpy(h_image_data.data(), d_image_buffer, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  // Create an OpenCV Mat from the host data. The data is already in RGB float format [0, 1].
  cv::Mat float_image(height, width, CV_32FC3, h_image_data.data());

  // Convert the float image to an 8-bit BGR image for saving
  cv::Mat bgr_image;
  float_image.convertTo(bgr_image, CV_8UC3, 255.0);
  cv::cvtColor(bgr_image, bgr_image, cv::COLOR_RGB2BGR);

  // Save the image to the specified file
  cv::imwrite(filename, bgr_image);
  std::cout << "Saved image to " << filename << std::endl;
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

void Trainer::reset_grad_accum(CudaDataManager &cuda) {
  CHECK_CUDA(cudaMemset(cuda.d_xyz_grad_accum, 0.0f, config.max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMemset(cuda.d_uv_grad_accum, 0.0f, config.max_gaussians * 2 * sizeof(float)));
  CHECK_CUDA(cudaMemset(cuda.d_grad_accum_dur, 0, config.max_gaussians * sizeof(int)));
}

void Trainer::reset_opacity(CudaDataManager &cuda) {
  // set all Gaussian opacity
  const double opc = config.reset_opacity_value;
  const float new_opc = log(opc) - log(1 - opc);

  CHECK_CUDA(cudaMemset(cuda.d_opacity, new_opc, config.max_gaussians * sizeof(float)));
}

void Trainer::add_sh_band(CudaDataManager &cuda) {}

void Trainer::adaptive_density(CudaDataManager &cuda, const int iter, const int num_gaussians, const int num_sh_coef) {}

float Trainer::backward_pass(const Image &curr_image, const Camera &curr_camera, CudaDataManager &cuda,
                             ForwardPassData &pass_data, const std::vector<cudaStream_t> &streams) {
  const int NUM_STREAMS = streams.size();
  const int width = (int)curr_camera.width;
  const int height = (int)curr_camera.height;

  // Load ground truth image and copy to device
  cv::Mat bgr_gt_image = cv::imread(curr_image.name, cv::IMREAD_COLOR);
  cv::Mat rgb_gt_image;
  cv::cvtColor(bgr_gt_image, rgb_gt_image, cv::COLOR_BGR2RGB);
  cv::Mat float_gt_image;
  rgb_gt_image.convertTo(float_gt_image, CV_32FC3, 1.0 / 255.0);
  std::vector<float> gt_image_data_vec((float *)float_gt_image.datastart, (float *)float_gt_image.dataend);

  float *d_gt_image;
  CHECK_CUDA(cudaMalloc(&d_gt_image, height * width * 3 * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(d_gt_image, gt_image_data_vec.data(), height * width * 3 * sizeof(float), cudaMemcpyHostToDevice));

  // Compute loss and image gradient
  float *d_grad_image;
  CHECK_CUDA(cudaMalloc(&d_grad_image, height * width * 3 * sizeof(float)));
  float loss = fused_loss(pass_data.d_image_buffer, d_gt_image, height, width, 3, config.ssim_frac, d_grad_image);

  // Backpropagate gradients from image to Gaussian parameters
  render_image_backward(cuda.d_uv_culled, cuda.d_opacity_culled, pass_data.d_conic, pass_data.d_precomputed_rgb, 1.0f,
                        pass_data.d_sorted_gaussians, pass_data.d_splat_start_end_idx_by_tile_idx,
                        pass_data.d_splats_per_pixel, pass_data.d_weight_per_pixel, d_grad_image, width, height,
                        cuda.d_grad_precompute_rgb, cuda.d_grad_opacity, cuda.d_grad_uv, cuda.d_grad_conic);

  int offset = 0;
  for (int i = 0; i < NUM_STREAMS; ++i) {
    int remainder = pass_data.num_culled % NUM_STREAMS;
    int size = pass_data.num_culled / NUM_STREAMS + (i < remainder ? 1 : 0);
    if (size <= 0)
      continue;

    const int num_sh_coef = (pass_data.l_max + 1) * (pass_data.l_max + 1) - 1;

    cudaStream_t stream = streams[i];
    precompute_spherical_harmonics_backward(cuda.d_xyz_c_culled + offset * 3, cuda.d_grad_precompute_rgb + offset * 3,
                                            pass_data.l_max, size, cuda.d_grad_sh + offset * num_sh_coef,
                                            cuda.d_grad_rgb + offset * 3, stream);
    compute_conic_backward(pass_data.d_J + offset * 6, pass_data.d_sigma + offset * 9, cuda.d_T,
                           cuda.d_grad_conic + offset * 3, size, cuda.d_grad_J + offset * 6,
                           cuda.d_grad_sigma + offset * 9, stream);
    compute_projection_jacobian_backward(cuda.d_xyz_c_culled + offset * 3, cuda.d_K, cuda.d_grad_J + offset * 6, size,
                                         cuda.d_grad_xyz_c + offset * 3, stream);
    compute_sigma_backward(cuda.d_quaternion_culled + offset * 4, cuda.d_scale_culled + offset * 3,
                           cuda.d_grad_sigma + offset * 9, size, cuda.d_grad_quaternion + offset * 4,
                           cuda.d_grad_scale + offset * 3, stream);
    camera_intrinsic_projection_backward(cuda.d_xyz_c_culled + offset * 3, cuda.d_K, cuda.d_grad_uv + offset * 2, size,
                                         cuda.d_grad_xyz_c + offset * 3, stream);
    camera_extrinsic_projection_backward(cuda.d_xyz_culled + offset * 3, cuda.d_T, cuda.d_grad_xyz_c + offset * 3, size,
                                         cuda.d_grad_xyz + offset * 3, stream);
    offset += size;
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaFree(d_gt_image));
  CHECK_CUDA(cudaFree(d_grad_image));

  return loss;
}

void Trainer::optimizer_step(CudaDataManager &cuda, const ForwardPassData &pass_data, const int iter,
                             const int num_gaussians, const int num_sh_coef) {
  // Select moment vectors by filtering based on the culled mask
  filter_moment_vectors(num_gaussians, 3, cuda.d_mask, cuda.m_grad_xyz, cuda.v_grad_xyz, cuda.m_grad_xyz_culled,
                        cuda.v_grad_xyz_culled);
  filter_moment_vectors(num_gaussians, 3, cuda.d_mask, cuda.m_grad_rgb, cuda.v_grad_rgb, cuda.m_grad_rgb_culled,
                        cuda.v_grad_rgb_culled);
  filter_moment_vectors(num_gaussians, 1, cuda.d_mask, cuda.m_grad_opacity, cuda.v_grad_opacity,
                        cuda.m_grad_opacity_culled, cuda.v_grad_opacity_culled);
  filter_moment_vectors(num_gaussians, 3, cuda.d_mask, cuda.m_grad_scale, cuda.v_grad_scale, cuda.m_grad_scale_culled,
                        cuda.v_grad_scale_culled);
  filter_moment_vectors(num_gaussians, 4, cuda.d_mask, cuda.m_grad_quaternion, cuda.v_grad_quaternion,
                        cuda.m_grad_quaternion_culled, cuda.v_grad_quaternion_culled);

  // Calculate bias correction terms for Adam optimizer
  float b1_t_corr = 1.0f - powf(B1, iter + 1);
  float b2_t_corr = 1.0f - powf(B2, iter + 1);

  // Update parameters using the Adam optimizer step
  adam_step(cuda.d_xyz_culled, cuda.d_grad_xyz, cuda.m_grad_xyz_culled, cuda.v_grad_xyz_culled,
            config.base_lr * config.xyz_lr_multiplier, B1, B2, EPS, b1_t_corr, b2_t_corr, pass_data.num_culled * 3);
  adam_step(cuda.d_rgb_culled, cuda.d_grad_rgb, cuda.m_grad_rgb_culled, cuda.v_grad_rgb_culled,
            config.base_lr * config.rgb_lr_multiplier, B1, B2, EPS, b1_t_corr, b2_t_corr, pass_data.num_culled * 3);
  adam_step(cuda.d_opacity_culled, cuda.d_grad_opacity, cuda.m_grad_opacity_culled, cuda.v_grad_opacity_culled,
            config.base_lr * config.opacity_lr_multiplier, B1, B2, EPS, b1_t_corr, b2_t_corr, pass_data.num_culled);
  adam_step(cuda.d_scale_culled, cuda.d_grad_scale, cuda.m_grad_scale_culled, cuda.v_grad_scale_culled,
            config.base_lr * config.scale_lr_multiplier, B1, B2, EPS, b1_t_corr, b2_t_corr, pass_data.num_culled * 3);
  adam_step(cuda.d_quaternion_culled, cuda.d_grad_quaternion, cuda.m_grad_quaternion_culled,
            cuda.v_grad_quaternion_culled, config.base_lr * config.quat_lr_multiplier, B1, B2, EPS, b1_t_corr,
            b2_t_corr, pass_data.num_culled * 4);

  // Scatter the updated moment vectors back to the full tensors
  scatter_params(num_gaussians, 3, cuda.d_mask, cuda.m_grad_xyz_culled, cuda.m_grad_xyz);
  scatter_params(num_gaussians, 3, cuda.d_mask, cuda.v_grad_xyz_culled, cuda.v_grad_xyz);
  scatter_params(num_gaussians, 3, cuda.d_mask, cuda.m_grad_rgb_culled, cuda.m_grad_rgb);
  scatter_params(num_gaussians, 3, cuda.d_mask, cuda.v_grad_rgb_culled, cuda.v_grad_rgb);
  scatter_params(num_gaussians, 1, cuda.d_mask, cuda.m_grad_opacity_culled, cuda.m_grad_opacity);
  scatter_params(num_gaussians, 1, cuda.d_mask, cuda.v_grad_opacity_culled, cuda.v_grad_opacity);
  scatter_params(num_gaussians, 3, cuda.d_mask, cuda.m_grad_scale_culled, cuda.m_grad_scale);
  scatter_params(num_gaussians, 3, cuda.d_mask, cuda.v_grad_scale_culled, cuda.v_grad_scale);
  scatter_params(num_gaussians, 4, cuda.d_mask, cuda.m_grad_quaternion_culled, cuda.m_grad_quaternion);
  scatter_params(num_gaussians, 4, cuda.d_mask, cuda.v_grad_quaternion_culled, cuda.v_grad_quaternion);

  // Scatter the updated parameters back to the full tensors
  scatter_params(num_gaussians, 3, cuda.d_mask, cuda.d_xyz_culled, cuda.d_xyz);
  scatter_params(num_gaussians, 3, cuda.d_mask, cuda.d_rgb_culled, cuda.d_rgb);
  scatter_params(num_gaussians, 1, cuda.d_mask, cuda.d_opacity_culled, cuda.d_opacity);
  scatter_params(num_gaussians, 3, cuda.d_mask, cuda.d_scale_culled, cuda.d_scale);
  scatter_params(num_gaussians, 4, cuda.d_mask, cuda.d_quaternion_culled, cuda.d_quaternion);

  // Update Spherical Harmonics (SH) parameters if they are being used
  if (num_sh_coef > 0) {
    filter_moment_vectors(num_gaussians, num_sh_coef, cuda.d_mask, cuda.m_grad_sh, cuda.v_grad_sh,
                          cuda.m_grad_sh_culled, cuda.v_grad_sh_culled);
    adam_step(cuda.d_sh_culled, cuda.d_grad_sh, cuda.m_grad_sh_culled, cuda.v_grad_sh_culled,
              config.base_lr * config.sh_lr_multiplier, B1, B2, EPS, b1_t_corr, b2_t_corr,
              pass_data.num_culled * num_sh_coef * 3);
    scatter_params(num_gaussians, num_sh_coef * 3, cuda.d_mask, cuda.m_grad_sh_culled, cuda.m_grad_sh);
    scatter_params(num_gaussians, num_sh_coef * 3, cuda.d_mask, cuda.v_grad_sh_culled, cuda.v_grad_sh);
    scatter_params(num_gaussians, num_sh_coef * 3, cuda.d_mask, cuda.d_sh_culled, cuda.d_sh);
  }

  // Update gradient accumulators
  accumulate_gradients(num_gaussians, cuda.d_mask, cuda.d_grad_xyz, cuda.d_grad_uv, cuda.d_xyz_grad_accum,
                       cuda.d_uv_grad_accum, cuda.d_grad_accum_dur);

  CHECK_CUDA(cudaDeviceSynchronize());
}
void Trainer::cleanup_iteration_buffers(ForwardPassData &pass_data) {
  CHECK_CUDA(cudaFree(pass_data.d_image_buffer));
  CHECK_CUDA(cudaFree(pass_data.d_weight_per_pixel));
  CHECK_CUDA(cudaFree(pass_data.d_splats_per_pixel));
  CHECK_CUDA(cudaFree(pass_data.d_sigma));
  CHECK_CUDA(cudaFree(pass_data.d_conic));
  CHECK_CUDA(cudaFree(pass_data.d_J));
  CHECK_CUDA(cudaFree(pass_data.d_splat_start_end_idx_by_tile_idx));
  CHECK_CUDA(cudaFree(pass_data.d_sorted_gaussians));
  CHECK_CUDA(cudaFree(pass_data.d_precomputed_rgb));
}

void Trainer::train() {
  constexpr int NUM_STREAMS = 4;

  // Setup: Initialize CUDA data manager, streams, and data splits
  CudaDataManager cuda(config.max_gaussians);
  std::vector<cudaStream_t> streams(NUM_STREAMS);
  for (int i = 0; i < NUM_STREAMS; ++i) {
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
  }
  test_train_split();
  reset_grad_accum(cuda);

  // Set optimizer moment vectors
  CHECK_CUDA(cudaMemset(cuda.m_grad_xyz, 0.0f, config.max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMemset(cuda.m_grad_rgb, 0.0f, config.max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMemset(cuda.m_grad_sh, 0.0f, config.max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMemset(cuda.m_grad_opacity, 0.0f, config.max_gaussians * sizeof(float)));
  CHECK_CUDA(cudaMemset(cuda.m_grad_scale, 0.0f, config.max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMemset(cuda.m_grad_quaternion, 0.0f, config.max_gaussians * 4 * sizeof(float)));
  CHECK_CUDA(cudaMemset(cuda.v_grad_xyz, 0.0f, config.max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMemset(cuda.v_grad_rgb, 0.0f, config.max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMemset(cuda.v_grad_sh, 0.0f, config.max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMemset(cuda.v_grad_opacity, 0.0f, config.max_gaussians * sizeof(float)));
  CHECK_CUDA(cudaMemset(cuda.v_grad_scale, 0.0f, config.max_gaussians * 3 * sizeof(float)));
  CHECK_CUDA(cudaMemset(cuda.v_grad_quaternion, 0.0f, config.max_gaussians * 4 * sizeof(float)));

  // TRAINING LOOP
  for (int iter = 0; iter < config.num_iters; ++iter) {
    std::cout << "ITER " << iter << std::endl;
    ForwardPassData pass_data;
    const int num_gaussians = gaussians.size();
    const int num_sh_coef = (pass_data.l_max + 1) * (pass_data.l_max + 1) - 1;

    // Zero out gradients
    CHECK_CUDA(cudaMemset(cuda.d_grad_xyz, 0.0f, 3 * num_gaussians * sizeof(float)));
    CHECK_CUDA(cudaMemset(cuda.d_grad_rgb, 0.0f, 3 * num_gaussians * sizeof(float)));
    CHECK_CUDA(cudaMemset(cuda.d_grad_sh, 0.0f, num_sh_coef * 3 * num_gaussians * sizeof(float)));
    CHECK_CUDA(cudaMemset(cuda.d_grad_opacity, 0.0f, num_gaussians * sizeof(float)));
    CHECK_CUDA(cudaMemset(cuda.d_grad_scale, 0.0f, 3 * num_gaussians * sizeof(float)));
    CHECK_CUDA(cudaMemset(cuda.d_grad_quaternion, 0.0f, 4 * num_gaussians * sizeof(float)));
    CHECK_CUDA(cudaMemset(cuda.d_grad_conic, 0.0f, 3 * num_gaussians * sizeof(float)));
    CHECK_CUDA(cudaMemset(cuda.d_grad_uv, 0.0f, 2 * num_gaussians * sizeof(float)));
    CHECK_CUDA(cudaMemset(cuda.d_grad_J, 0.0f, 6 * num_gaussians * sizeof(float)));
    CHECK_CUDA(cudaMemset(cuda.d_grad_sigma, 0.0f, 9 * num_gaussians * sizeof(float)));
    CHECK_CUDA(cudaMemset(cuda.d_grad_xyz_c, 0.0f, 3 * num_gaussians * sizeof(float)));
    CHECK_CUDA(cudaMemset(cuda.d_grad_precompute_rgb, 0.0f, 3 * num_gaussians * sizeof(float)));

    // Copy Gaussian data from host to device
    CHECK_CUDA(cudaMemcpy(cuda.d_xyz, gaussians.xyz.data(), num_gaussians * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(cuda.d_rgb, gaussians.rgb.data(), num_gaussians * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(cuda.d_opacity, gaussians.opacity.data(), num_gaussians * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(cuda.d_scale, gaussians.scale.data(), num_gaussians * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(cuda.d_quaternion, gaussians.quaternion.data(), num_gaussians * 4 * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Get current training image and camera
    Image curr_image = train_images[iter % train_images.size()];
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
    CHECK_CUDA(cudaMemcpy(cuda.d_K, h_K, 9 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(cuda.d_T, h_T, 12 * sizeof(float), cudaMemcpyHostToDevice));

    // --- FORWARD PASS via RASTERIZE MODULE ---
    rasterize_image(num_gaussians, curr_camera, config, cuda, pass_data, streams);

    if (pass_data.num_culled == 0) {
      std::cerr << "WARNING Image " << curr_image.id << " has no Gaussians in view" << std::endl;
      cleanup_iteration_buffers(pass_data);
      continue;
    }

    if (iter % 500 == 0)
      save_image(std::format("rendered_image_{}.png", iter), pass_data.d_image_buffer, curr_camera.width,
                 curr_camera.height);

    // --- BACKWARD PASS ---
    float loss = backward_pass(curr_image, curr_camera, cuda, pass_data, streams);
    std::cout << "LOSS TOTAL " << loss << std::endl;

    // --- OPTIMIZER STEP ---
    optimizer_step(cuda, pass_data, iter, num_gaussians, num_sh_coef);

    // Copy updated gaussians back to host
    CHECK_CUDA(cudaMemcpy(gaussians.xyz.data(), cuda.d_xyz, num_gaussians * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gaussians.rgb.data(), cuda.d_rgb, num_gaussians * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gaussians.opacity.data(), cuda.d_opacity, num_gaussians * 1 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(gaussians.scale.data(), cuda.d_scale, num_gaussians * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gaussians.quaternion.data(), cuda.d_quaternion, num_gaussians * 4 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // --- ADAPTIVE DENSITY ---
    if (iter > config.adaptive_control_start && iter % config.adaptive_control_interval == 0 &&
        iter < config.adaptive_control_end) {
      adaptive_density(cuda, iter, num_gaussians, num_sh_coef);
      reset_grad_accum(cuda);
    }

    if (iter > config.reset_opacity_start && iter % config.reset_opacity_interval == 0 &&
        iter < config.reset_opacity_end) {
      reset_opacity(cuda);
      reset_grad_accum(cuda);
    }

    // Free temporary buffers for this iteration
    cleanup_iteration_buffers(pass_data);
  }
}
