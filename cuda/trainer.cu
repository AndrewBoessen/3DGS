// trainer.cu

#include "gsplat/trainer.hpp"

#include "gsplat/progress_bar.hpp"
#include "gsplat_cuda/adaptive_density.cuh"
#include "gsplat_cuda/cuda_backward.cuh"
#include "gsplat_cuda/cuda_data.cuh"
#include "gsplat_cuda/cuda_forward.cuh"
#include "gsplat_cuda/optimizer.cuh"
#include "gsplat_cuda/raster.cuh"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <random>
#include <thread>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

/**
 * @brief Private implementation (PImpl) for the Trainer class.
 *
 * This class holds all the state and CUDA-specific data and logic.
 */
class TrainerImpl {
public:
  // --- Constructor ---
  TrainerImpl(ConfigParameters config_in, Gaussians gaussians_in, std::unordered_map<int, Image> images_in,
              std::unordered_map<int, Camera> cameras_in)
      : config(std::move(config_in)), gaussians(std::move(gaussians_in)), images(std::move(images_in)),
        cameras(std::move(cameras_in)), iter(0), l_max(0), num_gaussians(gaussians.size()),
        cuda(config.max_gaussians) // Initialize the CUDA data manager
  {}

  // --- Public methods (called by Trainer) ---
  void test_train_split();
  void train();
  void evaluate();

private:
  // --- All original private members ---
  ConfigParameters config;
  Gaussians gaussians;
  std::unordered_map<int, Image> images;
  std::unordered_map<int, Camera> cameras;

  std::vector<Image> test_images;
  std::vector<Image> train_images;

  int iter;
  int l_max;
  int num_gaussians;
  float scene_extent;

  // --- CUDA-specific data ---
  CudaDataManager cuda;

  void reset_grad_accum();
  void reset_opacity();
  void zero_grads();
  float backward_pass(const Image &curr_image, const Camera &curr_camera, ForwardPassData &pass_data,
                      const float bg_color, const thrust::device_vector<float> &d_gt_image);
  void optimizer_step(ForwardPassData pass_data);
  void add_sh_band();
  void adaptive_density_step();

  // --- Async Loading Members ---
  std::thread loader_thread;
  std::mutex load_mutex;
  std::condition_variable load_cv;

  // Flags for synchronization
  bool buffer_ready[2] = {false, false}; // Is the host buffer filled?
  bool shutdown_requested = false;
  bool request_load[2] = {false, false}; // Request to load into specific buffer

  // Data for the loader to know what to fetch
  int next_image_index = -1;
  int current_load_buffer_idx = 0; // Which buffer is the loader currently filling?

  // Pinned Memory Buffers (Host side) - Double Buffered
  float *h_pinned_image_buffer[2] = {nullptr, nullptr};
  size_t pinned_buffer_size = 0;

  // Device buffers to hold the GT images - Double Buffered
  thrust::device_vector<float> d_gt_image[2];
  int buffer_image_indices[2] = {-1, -1}; // Tracks which image index is in each buffer

  // RNG
  std::mt19937 rng;

  // CUDA Stream and Event for Async Transfer
  cudaStream_t transfer_stream;
  cudaEvent_t copy_finished_events[2];

  // Helper to initialize pinned memory
  void init_pinned_memory(size_t max_pixels) {
    // Allocate enough for largest image * 3 channels * sizeof(float)
    pinned_buffer_size = max_pixels * 3 * sizeof(float);
    cudaMallocHost((void **)&h_pinned_image_buffer[0], pinned_buffer_size);
    cudaMallocHost((void **)&h_pinned_image_buffer[1], pinned_buffer_size);

    cudaStreamCreate(&transfer_stream);
    cudaEventCreate(&copy_finished_events[0]);
    cudaEventCreate(&copy_finished_events[1]);
  }

  // The loop that runs in the background thread
  void image_loader_loop();

  // Clean up
  void free_pinned_memory() {
    if (h_pinned_image_buffer[0]) {
      cudaFreeHost(h_pinned_image_buffer[0]);
      h_pinned_image_buffer[0] = nullptr;
    }
    if (h_pinned_image_buffer[1]) {
      cudaFreeHost(h_pinned_image_buffer[1]);
      h_pinned_image_buffer[1] = nullptr;
    }
    cudaStreamDestroy(transfer_stream);
    cudaEventDestroy(copy_finished_events[0]);
    cudaEventDestroy(copy_finished_events[1]);
  }
};

void TrainerImpl::image_loader_loop() {
  while (true) {
    std::unique_lock<std::mutex> lock(load_mutex);

    // Wait until main thread requests a load or shuts down
    // We check if ANY buffer needs loading
    load_cv.wait(lock, [this] { return request_load[0] || request_load[1] || shutdown_requested; });

    if (shutdown_requested)
      break;

    // Determine which buffer to load
    int buf_idx = -1;
    if (request_load[0])
      buf_idx = 0;
    else if (request_load[1])
      buf_idx = 1;

    if (buf_idx == -1)
      continue; // Should not happen

    // Grab the index to load
    // Note: In a real double buffering scenario, we'd need to know WHICH image index goes to WHICH buffer.
    // For simplicity, we'll assume the main thread sets 'next_image_index' correctly before requesting.
    // BUT, since we might pipeline, we need to be careful.
    // Let's assume the main thread sets 'next_image_index' for the *current* request.
    // Actually, to be robust, we should probably have a queue or array of indices.
    // For now, let's assume the main thread manages the index and we just load what's asked.
    // Wait, if we want to load N+2 while N is training, we need to know N+2.
    // Let's assume the main thread updates 'next_image_index' and then sets 'request_load[buf_idx]'.

    int img_idx = next_image_index;
    request_load[buf_idx] = false; // Acknowledge request
    lock.unlock();                 // Release lock while doing heavy I/O

    // 1. Disk I/O
    const Image &next_img = train_images[img_idx];
    cv::Mat bgr_image = cv::imread(next_img.name, cv::IMREAD_COLOR);

    // Validation (basic)
    if (bgr_image.empty()) {
      std::cerr << "Async Error: Failed to load " << next_img.name << std::endl;
      memset(h_pinned_image_buffer[buf_idx], 0, pinned_buffer_size);
    } else {
      // 2. CPU Compute (Color conversion & Float cast)
      cv::Mat rgb_image;
      cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

      cv::Mat float_image;
      rgb_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

      // 3. Write to Pinned Memory
      size_t copy_size = float_image.total() * float_image.elemSize();
      if (copy_size <= pinned_buffer_size) {
        memcpy(h_pinned_image_buffer[buf_idx], float_image.ptr<float>(0), copy_size);
      }
    }

    // Notify main thread that data is ready
    lock.lock();
    buffer_ready[buf_idx] = true;
    lock.unlock();
    load_cv.notify_all();
  }
}

// --- Implementation of TrainerImpl methods ---

void TrainerImpl::test_train_split() {
  const int split = config.test_split_ratio;

  test_images.clear();
  train_images.clear();

  if (images.empty()) {
    return;
  }

  std::vector<Image> all_images;
  all_images.reserve(images.size());
  for (const auto &pair : images) {
    all_images.push_back(pair.second);
  }

  std::sort(all_images.begin(), all_images.end(), [](const Image &a, const Image &b) { return a.name < b.name; });

  if (split <= 0) {
    train_images = all_images;
  } else {
    for (size_t i = 0; i < all_images.size(); ++i) {
      if (i % split == 0) {
        test_images.push_back(all_images[i]);
      } else {
        train_images.push_back(all_images[i]);
      }
    }
  }
}

void TrainerImpl::reset_grad_accum() {
  thrust::fill_n(cuda.accumulators.d_uv_grad_accum.begin(), num_gaussians, 0.0f);
  thrust::fill_n(cuda.accumulators.d_grad_accum_dur.begin(), num_gaussians, 0);
}

void TrainerImpl::reset_opacity() {
  const double opc = config.reset_opacity_value;
  const float new_opc = log(opc) - log(1.0f - opc);

  thrust::fill_n(cuda.gaussians.d_opacity.begin(), num_gaussians, new_opc);
  thrust::fill_n(cuda.optimizer.m_grad_opacity.begin(), num_gaussians, 0.0f);
  thrust::fill_n(cuda.optimizer.v_grad_opacity.begin(), num_gaussians, 0.0f);
}

void TrainerImpl::zero_grads() {
  thrust::fill_n(cuda.gradients.d_grad_xyz.begin(), num_gaussians * 3, 0.0f);
  thrust::fill_n(cuda.gradients.d_grad_rgb.begin(), num_gaussians * 3, 0.0f);
  thrust::fill_n(cuda.gradients.d_grad_sh.begin(), num_gaussians * 15 * 3, 0.0f);
  thrust::fill_n(cuda.gradients.d_grad_opacity.begin(), num_gaussians, 0.0f);
  thrust::fill_n(cuda.gradients.d_grad_scale.begin(), num_gaussians * 3, 0.0f);
  thrust::fill_n(cuda.gradients.d_grad_quaternion.begin(), num_gaussians * 4, 0.0f);

  thrust::fill_n(cuda.gradients.d_grad_conic.begin(), num_gaussians * 3, 0.0f);
  thrust::fill_n(cuda.gradients.d_grad_uv.begin(), num_gaussians * 2, 0.0f);
  thrust::fill_n(cuda.gradients.d_grad_J.begin(), num_gaussians * 6, 0.0f);
  thrust::fill_n(cuda.gradients.d_grad_sigma.begin(), num_gaussians * 9, 0.0f);
  thrust::fill_n(cuda.gradients.d_grad_xyz_c.begin(), num_gaussians * 3, 0.0f);
  thrust::fill_n(cuda.gradients.d_grad_precompute_rgb.begin(), num_gaussians * 3, 0.0f);
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

void TrainerImpl::add_sh_band() {
  if (l_max >= config.max_sh_band)
    return;

  if (l_max == 0) {
    thrust::fill(cuda.gaussians.d_sh.begin(), cuda.gaussians.d_sh.end(), 0.0f);
    l_max++;
    return;
  }
  const int curr_param_count = (l_max + 1) * (l_max + 1) - 1;
  const int new_param_count = (l_max + 2) * (l_max + 2) - 1;

  const int old_coeffs_per_gaussian = curr_param_count * 3;
  const int new_coeffs_per_gaussian = new_param_count * 3;

  try {
    thrust::device_vector<float> temp_sh(num_gaussians * old_coeffs_per_gaussian);
    thrust::copy_n(cuda.gaussians.d_sh.begin(), temp_sh.size(), temp_sh.begin());

    thrust::device_vector<int> destination_map(temp_sh.size());

    SHIndexMapper mapper(old_coeffs_per_gaussian, new_coeffs_per_gaussian);

    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(temp_sh.size()),
                      destination_map.begin(), mapper);

    thrust::fill(cuda.gaussians.d_sh.begin(), cuda.gaussians.d_sh.end(), 0.0f);

    thrust::scatter(temp_sh.begin(), temp_sh.end(), destination_map.begin(), cuda.gaussians.d_sh.begin());

  } catch (const std::exception &e) {
    fprintf(stderr, "Error during SH band expansion: %s\n", e.what());
    exit(EXIT_FAILURE);
  }

  l_max++;
}

// Computes the L2 norm of the average 2D gradient.
struct ComputeAvgGrad {
  __host__ __device__ float operator()(const thrust::tuple<float, int> &t) const {
    float grad_sum = thrust::get<0>(t);
    int duration = thrust::get<1>(t);

    if (duration == 0)
      return 0.0f;
    return grad_sum / (float)duration;
  }
};

// Get max of scale parameters
struct ComputeScaleMax {
  __host__ __device__ float operator()(const thrust::tuple<float, float, float> &t) const {
    float s1 = thrust::get<0>(t);
    float s2 = thrust::get<1>(t);
    float s3 = thrust::get<2>(t);

    return fmaxf(expf(s1), fmaxf(expf(s2), expf(s3)));
  }
};

// Identifies Gaussians to be pruned based on low opacity or large scale.
struct IdentifyPrune {
  const float op_threshold;
  const float scale_max;

  IdentifyPrune(float ot, float sm) : op_threshold(ot), scale_max(sm) {}

  __host__ __device__ bool operator()(const thrust::tuple<float, float> &t) const {
    float opacity_logit = thrust::get<0>(t);
    float scale = thrust::get<1>(t);

    // Prune if opacity is too low
    if (opacity_logit < op_threshold)
      return true;

    // Prune if scale is too large
    if (scale > scale_max)
      return true;

    return false;
  }
};

// Identifies Gaussians to be cloned based on high gradient.
struct IdentifyClone {
  const float grad_threshold;
  const float scale_threshold;
  IdentifyClone(float gt, float st) : grad_threshold(gt), scale_threshold(st) {}

  __host__ __device__ bool operator()(const thrust::tuple<float, float, bool> &t) const {
    float avg_grad = thrust::get<0>(t);
    float scale = thrust::get<1>(t);
    bool is_pruned = thrust::get<2>(t);

    if (is_pruned)
      return false;
    return avg_grad > grad_threshold && scale <= scale_threshold;
  }
};

// Identifies Gaussians to be split based on high gradient.
struct IdentifySplit {
  const float grad_threshold;
  const float scale_threshold;
  IdentifySplit(float gt, float st) : grad_threshold(gt), scale_threshold(st) {}

  __host__ __device__ bool operator()(const thrust::tuple<float, float, bool> &t) const {
    float avg_grad = thrust::get<0>(t);
    float scale = thrust::get<1>(t);
    bool is_pruned = thrust::get<2>(t);

    if (is_pruned)
      return false;
    return avg_grad > grad_threshold && scale > scale_threshold;
  }
};

// Combines prune and split masks into a single keep mask.
struct CombineMasks {
  __host__ __device__ bool operator()(const thrust::tuple<bool, bool, bool> &t) const {
    return !(thrust::get<0>(t) || // is_pruned
             thrust::get<1>(t) || // is_split
             thrust::get<2>(t));  // is_clone
  }
};

struct BoolToInt {
  __host__ __device__ int operator()(bool x) { return x ? 1 : 0; }
};

void TrainerImpl::adaptive_density_step() {
  // --- 1. Calculate Average Gradient Norms and Scale Max---
  thrust::device_vector<float> d_avg_uv_grad_norm(num_gaussians);

  auto avg_uv_grad_iter_start = thrust::make_zip_iterator(
      thrust::make_tuple(cuda.accumulators.d_uv_grad_accum.begin(), cuda.accumulators.d_grad_accum_dur.begin()));
  auto avg_uv_grad_iter_end = avg_uv_grad_iter_start + num_gaussians;

  thrust::transform(avg_uv_grad_iter_start, avg_uv_grad_iter_end, d_avg_uv_grad_norm.begin(), ComputeAvgGrad());

  thrust::device_vector<float> d_scale_max(num_gaussians);

  auto s1_it = thrust::make_strided_iterator(cuda.gaussians.d_scale.begin(), 3);
  auto s2_it = thrust::make_strided_iterator(cuda.gaussians.d_scale.begin() + 1, 3);
  auto s3_it = thrust::make_strided_iterator(cuda.gaussians.d_scale.begin() + 2, 3);

  auto scale_iter_start = thrust::make_zip_iterator(thrust::make_tuple(s1_it, s2_it, s3_it));
  auto scale_iter_end = scale_iter_start + num_gaussians;

  thrust::transform(scale_iter_start, scale_iter_end, d_scale_max.begin(), ComputeScaleMax());

  const float max_scale = scene_extent * 0.1f;
  const float clone_scale_threshold = scene_extent * 0.01f;

  // --- 2. Identify Gaussians to Prune ---
  // Inverse sigmoid: log(p / (1-p))
  const float op_threshold = logf(config.delete_opacity_threshold) - logf(1.0f - config.delete_opacity_threshold);

  auto prune_iter_start =
      thrust::make_zip_iterator(thrust::make_tuple(cuda.gaussians.d_opacity.begin(), d_scale_max.begin()));
  auto prune_iter_end = prune_iter_start + num_gaussians;

  thrust::device_vector<bool> d_prune_mask(num_gaussians);
  thrust::transform(prune_iter_start, prune_iter_end, d_prune_mask.begin(), IdentifyPrune(op_threshold, max_scale));

  int num_to_prune = thrust::count(d_prune_mask.begin(), d_prune_mask.end(), true);

  // --- 3. Identify Gaussians to Clone ---
  auto densify_iter_start = thrust::make_zip_iterator(
      thrust::make_tuple(d_avg_uv_grad_norm.begin(), d_scale_max.begin(), d_prune_mask.begin()));
  auto densify_iter_end = densify_iter_start + num_gaussians;

  thrust::device_vector<bool> d_clone_mask(num_gaussians);
  thrust::transform(densify_iter_start, densify_iter_end, d_clone_mask.begin(),
                    IdentifyClone(config.uv_grad_threshold, clone_scale_threshold));

  int num_to_clone = thrust::count(d_clone_mask.begin(), d_clone_mask.end(), true);

  // --- 4. Identify Gaussians to Split ---
  thrust::device_vector<bool> d_split_mask(num_gaussians);
  thrust::transform(densify_iter_start, densify_iter_end, d_split_mask.begin(),
                    IdentifySplit(config.uv_grad_threshold, clone_scale_threshold));

  int num_to_split = thrust::count(d_split_mask.begin(), d_split_mask.end(), true);

  // --- 5. Check Capacity ---
  int num_to_remove = num_to_prune + num_to_split + num_to_clone;
  int num_to_add = (num_to_clone * 2) + (num_to_split * 2);
  int new_num_gaussians = num_gaussians - num_to_remove + num_to_add;

  if (new_num_gaussians > config.max_gaussians) {
    std::cerr << "WARNING: Adaptive density step would exceed max_gaussians (" << new_num_gaussians << " > "
              << config.max_gaussians << "). Skipping." << std::endl;
    // TODO: A more robust strategy would be to prune anyway, and then fill
    // remaining capacity with the highest-gradient clones/splits.
    return;
  }

  if (num_to_add == 0 && num_to_prune == 0) {
    return; // Nothing to do
  }

  // --- 6. Generate New Gaussian Parameters (Kernels) ---
  const int num_sh_coeffs = (l_max > 0) ? ((l_max + 1) * (l_max + 1) - 1) : 0;

  // Allocate temp device memory for new Gaussians
  thrust::device_vector<float> d_new_clone_xyz(num_to_clone * 2 * 3);
  thrust::device_vector<float> d_new_clone_rgb(num_to_clone * 2 * 3);
  thrust::device_vector<float> d_new_clone_opacity(num_to_clone * 2 * 1);
  thrust::device_vector<float> d_new_clone_scale(num_to_clone * 2 * 3);
  thrust::device_vector<float> d_new_clone_quat(num_to_clone * 2 * 4);
  thrust::device_vector<float> d_new_clone_sh(num_to_clone * 2 * num_sh_coeffs * 3);

  thrust::device_vector<float> d_new_split_xyz(num_to_split * 2 * 3);
  thrust::device_vector<float> d_new_split_rgb(num_to_split * 2 * 3);
  thrust::device_vector<float> d_new_split_opacity(num_to_split * 2 * 1);
  thrust::device_vector<float> d_new_split_scale(num_to_split * 2 * 3);
  thrust::device_vector<float> d_new_split_quat(num_to_split * 2 * 4);
  thrust::device_vector<float> d_new_split_sh(num_to_split * 2 * num_sh_coeffs * 3);

  if (num_to_clone > 0) {
    thrust::device_vector<int> clone_write_ids(num_gaussians);
    auto clone_int_mask_start = thrust::make_transform_iterator(d_clone_mask.begin(), BoolToInt());
    auto clone_int_mask_end = thrust::make_transform_iterator(d_clone_mask.end(), BoolToInt());
    thrust::exclusive_scan(clone_int_mask_start, clone_int_mask_end, clone_write_ids.begin());
    clone_gaussians(
        num_gaussians, num_sh_coeffs, thrust::raw_pointer_cast(d_clone_mask.data()),
        thrust::raw_pointer_cast(clone_write_ids.data()), thrust::raw_pointer_cast(cuda.gaussians.d_xyz.data()),
        thrust::raw_pointer_cast(cuda.gaussians.d_rgb.data()),
        thrust::raw_pointer_cast(cuda.gaussians.d_opacity.data()),
        thrust::raw_pointer_cast(cuda.gaussians.d_scale.data()),
        thrust::raw_pointer_cast(cuda.gaussians.d_quaternion.data()),
        thrust::raw_pointer_cast(cuda.gaussians.d_sh.data()), thrust::raw_pointer_cast(d_new_clone_xyz.data()),
        thrust::raw_pointer_cast(d_new_clone_rgb.data()), thrust::raw_pointer_cast(d_new_clone_opacity.data()),
        thrust::raw_pointer_cast(d_new_clone_scale.data()), thrust::raw_pointer_cast(d_new_clone_quat.data()),
        thrust::raw_pointer_cast(d_new_clone_sh.data()));
  }

  if (num_to_split > 0) {
    thrust::device_vector<int> split_write_ids(num_gaussians);
    auto split_int_mask_start = thrust::make_transform_iterator(d_split_mask.begin(), BoolToInt());
    auto split_int_mask_end = thrust::make_transform_iterator(d_split_mask.end(), BoolToInt());
    thrust::exclusive_scan(split_int_mask_start, split_int_mask_end, split_write_ids.begin());
    split_gaussians(
        num_gaussians, config.split_scale_factor, num_sh_coeffs, thrust::raw_pointer_cast(d_split_mask.data()),
        thrust::raw_pointer_cast(split_write_ids.data()), thrust::raw_pointer_cast(cuda.gaussians.d_xyz.data()),
        thrust::raw_pointer_cast(cuda.gaussians.d_rgb.data()),
        thrust::raw_pointer_cast(cuda.gaussians.d_opacity.data()),
        thrust::raw_pointer_cast(cuda.gaussians.d_scale.data()),
        thrust::raw_pointer_cast(cuda.gaussians.d_quaternion.data()),
        thrust::raw_pointer_cast(cuda.gaussians.d_sh.data()), thrust::raw_pointer_cast(d_new_split_xyz.data()),
        thrust::raw_pointer_cast(d_new_split_rgb.data()), thrust::raw_pointer_cast(d_new_split_opacity.data()),
        thrust::raw_pointer_cast(d_new_split_scale.data()), thrust::raw_pointer_cast(d_new_split_quat.data()),
        thrust::raw_pointer_cast(d_new_split_sh.data()));
  }

  // --- 8, 9, 10. Compact existing vectors and append new data ---
  // - Compact all params in keep mask
  // - Fill front with compact params
  // - Append Clone and Split params to back
  thrust::device_vector<bool> d_keep_mask(num_gaussians);
  auto keep_mask_start =
      thrust::make_zip_iterator(thrust::make_tuple(d_prune_mask.begin(), d_split_mask.begin(), d_clone_mask.begin()));
  auto keep_mask_end = keep_mask_start + num_gaussians;
  thrust::transform(keep_mask_start, keep_mask_end, d_keep_mask.begin(), CombineMasks());

  const int keep_size = thrust::count(d_keep_mask.begin(), d_keep_mask.end(), true);

  // Select keep params
  auto keep_xyz = compact_masked_array<3>(cuda.gaussians.d_xyz, d_keep_mask, keep_size);
  auto keep_rgb = compact_masked_array<3>(cuda.gaussians.d_rgb, d_keep_mask, keep_size);
  auto keep_op = compact_masked_array<1>(cuda.gaussians.d_opacity, d_keep_mask, keep_size);
  auto keep_scale = compact_masked_array<3>(cuda.gaussians.d_scale, d_keep_mask, keep_size);
  auto keep_quat = compact_masked_array<4>(cuda.gaussians.d_quaternion, d_keep_mask, keep_size);

  // Select keep optimizer states
  auto keep_xyz_m = compact_masked_array<3>(cuda.optimizer.m_grad_xyz, d_keep_mask, keep_size);
  auto keep_rgb_m = compact_masked_array<3>(cuda.optimizer.m_grad_rgb, d_keep_mask, keep_size);
  auto keep_op_m = compact_masked_array<1>(cuda.optimizer.m_grad_opacity, d_keep_mask, keep_size);
  auto keep_scale_m = compact_masked_array<3>(cuda.optimizer.m_grad_scale, d_keep_mask, keep_size);
  auto keep_quat_m = compact_masked_array<4>(cuda.optimizer.m_grad_quaternion, d_keep_mask, keep_size);

  auto keep_xyz_v = compact_masked_array<3>(cuda.optimizer.v_grad_xyz, d_keep_mask, keep_size);
  auto keep_rgb_v = compact_masked_array<3>(cuda.optimizer.v_grad_rgb, d_keep_mask, keep_size);
  auto keep_op_v = compact_masked_array<1>(cuda.optimizer.v_grad_opacity, d_keep_mask, keep_size);
  auto keep_scale_v = compact_masked_array<3>(cuda.optimizer.v_grad_scale, d_keep_mask, keep_size);
  auto keep_quat_v = compact_masked_array<4>(cuda.optimizer.v_grad_quaternion, d_keep_mask, keep_size);

  // Select SH params and optimizer states
  thrust::device_vector<float> keep_sh;
  thrust::device_vector<float> keep_sh_m;
  thrust::device_vector<float> keep_sh_v;
  switch (l_max) {
  case 0:
    break;
  case 1:
    keep_sh = compact_masked_array<9>(cuda.gaussians.d_sh, d_keep_mask, keep_size);
    keep_sh_m = compact_masked_array<9>(cuda.optimizer.m_grad_sh, d_keep_mask, keep_size);
    keep_sh_v = compact_masked_array<9>(cuda.optimizer.v_grad_sh, d_keep_mask, keep_size);
    break;
  case 2:
    keep_sh = compact_masked_array<24>(cuda.gaussians.d_sh, d_keep_mask, keep_size);
    keep_sh_m = compact_masked_array<24>(cuda.optimizer.m_grad_sh, d_keep_mask, keep_size);
    keep_sh_v = compact_masked_array<24>(cuda.optimizer.v_grad_sh, d_keep_mask, keep_size);
    break;
  case 3:
    keep_sh = compact_masked_array<45>(cuda.gaussians.d_sh, d_keep_mask, keep_size);
    keep_sh_m = compact_masked_array<45>(cuda.optimizer.m_grad_sh, d_keep_mask, keep_size);
    keep_sh_v = compact_masked_array<45>(cuda.optimizer.v_grad_sh, d_keep_mask, keep_size);
    break;
  default:
    fprintf(stderr, "Error SH band is invalid\n");
    exit(EXIT_FAILURE);
  }
  // Zero out all optimizer states
  thrust::fill(cuda.optimizer.m_grad_xyz.begin(), cuda.optimizer.m_grad_xyz.end(), 0.0f);
  thrust::fill(cuda.optimizer.m_grad_rgb.begin(), cuda.optimizer.m_grad_rgb.end(), 0.0f);
  thrust::fill(cuda.optimizer.m_grad_opacity.begin(), cuda.optimizer.m_grad_opacity.end(), 0.0f);
  thrust::fill(cuda.optimizer.m_grad_scale.begin(), cuda.optimizer.m_grad_scale.end(), 0.0f);
  thrust::fill(cuda.optimizer.m_grad_quaternion.begin(), cuda.optimizer.m_grad_quaternion.end(), 0.0f);
  thrust::fill(cuda.optimizer.m_grad_sh.begin(), cuda.optimizer.m_grad_sh.end(), 0.0f);

  thrust::fill(cuda.optimizer.v_grad_xyz.begin(), cuda.optimizer.v_grad_xyz.end(), 0.0f);
  thrust::fill(cuda.optimizer.v_grad_rgb.begin(), cuda.optimizer.v_grad_rgb.end(), 0.0f);
  thrust::fill(cuda.optimizer.v_grad_opacity.begin(), cuda.optimizer.v_grad_opacity.end(), 0.0f);
  thrust::fill(cuda.optimizer.v_grad_scale.begin(), cuda.optimizer.v_grad_scale.end(), 0.0f);
  thrust::fill(cuda.optimizer.v_grad_quaternion.begin(), cuda.optimizer.v_grad_quaternion.end(), 0.0f);
  thrust::fill(cuda.optimizer.v_grad_sh.begin(), cuda.optimizer.v_grad_sh.end(), 0.0f);

  // Fill with kept Gaussians
  thrust::copy(keep_xyz.begin(), keep_xyz.end(), cuda.gaussians.d_xyz.begin());
  thrust::copy(keep_rgb.begin(), keep_rgb.end(), cuda.gaussians.d_rgb.begin());
  thrust::copy(keep_op.begin(), keep_op.end(), cuda.gaussians.d_opacity.begin());
  thrust::copy(keep_scale.begin(), keep_scale.end(), cuda.gaussians.d_scale.begin());
  thrust::copy(keep_quat.begin(), keep_quat.end(), cuda.gaussians.d_quaternion.begin());

  thrust::copy(keep_xyz_m.begin(), keep_xyz_m.end(), cuda.optimizer.m_grad_xyz.begin());
  thrust::copy(keep_rgb_m.begin(), keep_rgb_m.end(), cuda.optimizer.m_grad_rgb.begin());
  thrust::copy(keep_op_m.begin(), keep_op_m.end(), cuda.optimizer.m_grad_opacity.begin());
  thrust::copy(keep_scale_m.begin(), keep_scale_m.end(), cuda.optimizer.m_grad_scale.begin());
  thrust::copy(keep_quat_m.begin(), keep_quat_m.end(), cuda.optimizer.m_grad_quaternion.begin());

  thrust::copy(keep_xyz_v.begin(), keep_xyz_v.end(), cuda.optimizer.v_grad_xyz.begin());
  thrust::copy(keep_rgb_v.begin(), keep_rgb_v.end(), cuda.optimizer.v_grad_rgb.begin());
  thrust::copy(keep_op_v.begin(), keep_op_v.end(), cuda.optimizer.v_grad_opacity.begin());
  thrust::copy(keep_scale_v.begin(), keep_scale_v.end(), cuda.optimizer.v_grad_scale.begin());
  thrust::copy(keep_quat_v.begin(), keep_quat_v.end(), cuda.optimizer.v_grad_quaternion.begin());

  if (l_max > 0) {
    thrust::copy(keep_sh.begin(), keep_sh.end(), cuda.gaussians.d_sh.begin());
    thrust::copy(keep_sh_m.begin(), keep_sh_m.end(), cuda.optimizer.m_grad_sh.begin());
    thrust::copy(keep_sh_v.begin(), keep_sh_v.end(), cuda.optimizer.v_grad_sh.begin());
  }

  // Fill back with new cloned and split Gaussians
  thrust::copy(d_new_clone_xyz.begin(), d_new_clone_xyz.end(), cuda.gaussians.d_xyz.begin() + keep_size * 3);
  thrust::copy(d_new_clone_rgb.begin(), d_new_clone_rgb.end(), cuda.gaussians.d_rgb.begin() + keep_size * 3);
  thrust::copy(d_new_clone_opacity.begin(), d_new_clone_opacity.end(), cuda.gaussians.d_opacity.begin() + keep_size);
  thrust::copy(d_new_clone_scale.begin(), d_new_clone_scale.end(), cuda.gaussians.d_scale.begin() + keep_size * 3);
  thrust::copy(d_new_clone_quat.begin(), d_new_clone_quat.end(), cuda.gaussians.d_quaternion.begin() + keep_size * 4);

  thrust::copy(d_new_split_xyz.begin(), d_new_split_xyz.end(),
               cuda.gaussians.d_xyz.begin() + (keep_size + num_to_clone * 2) * 3);
  thrust::copy(d_new_split_rgb.begin(), d_new_split_rgb.end(),
               cuda.gaussians.d_rgb.begin() + (keep_size + num_to_clone * 2) * 3);
  thrust::copy(d_new_split_opacity.begin(), d_new_split_opacity.end(),
               cuda.gaussians.d_opacity.begin() + (keep_size + num_to_clone * 2));
  thrust::copy(d_new_split_scale.begin(), d_new_split_scale.end(),
               cuda.gaussians.d_scale.begin() + (keep_size + num_to_clone * 2) * 3);
  thrust::copy(d_new_split_quat.begin(), d_new_split_quat.end(),
               cuda.gaussians.d_quaternion.begin() + (keep_size + num_to_clone * 2) * 4);

  if (l_max > 0) {
    thrust::copy(d_new_clone_sh.begin(), d_new_clone_sh.end(),
                 cuda.gaussians.d_sh.begin() + keep_size * num_sh_coeffs * 3);
    thrust::copy(d_new_split_sh.begin(), d_new_split_sh.end(),
                 cuda.gaussians.d_sh.begin() + (keep_size + num_to_clone * 2) * num_sh_coeffs * 3);
  }

  // --- 11. Update total Gaussian count ---
  num_gaussians = keep_size + (num_to_clone + num_to_split) * 2;

  if (num_gaussians != new_num_gaussians) {
    std::cerr << "FATAL ERROR: Gaussian count mismatch in adaptive density!" << std::endl;
    exit(EXIT_FAILURE);
  }
}

float TrainerImpl::backward_pass(const Image &curr_image, const Camera &curr_camera, ForwardPassData &pass_data,
                                 const float bg_color, const thrust::device_vector<float> &d_gt_image) {
  const int width = (int)curr_camera.width;
  const int height = (int)curr_camera.height;

  thrust::device_vector<float> d_grad_image(height * width * 3);

  float loss =
      fused_loss(thrust::raw_pointer_cast(pass_data.d_image_buffer.data()), thrust::raw_pointer_cast(d_gt_image.data()),
                 height, width, config.ssim_frac, thrust::raw_pointer_cast(d_grad_image.data()));

  auto d_uv_selected = compact_masked_array<2>(pass_data.d_uv, pass_data.d_mask, pass_data.num_culled);
  auto d_opacity_selected = compact_masked_array<1>(cuda.gaussians.d_opacity, pass_data.d_mask, pass_data.num_culled);
  auto d_xyz_c_selected = compact_masked_array<3>(pass_data.d_xyz_c, pass_data.d_mask, pass_data.num_culled);
  auto d_quaternion_selected =
      compact_masked_array<4>(cuda.gaussians.d_quaternion, pass_data.d_mask, pass_data.num_culled);
  auto d_scale_selected = compact_masked_array<3>(cuda.gaussians.d_scale, pass_data.d_mask, pass_data.num_culled);
  auto d_xyz_selected = compact_masked_array<3>(cuda.gaussians.d_xyz, pass_data.d_mask, pass_data.num_culled);

  render_image_backward(
      thrust::raw_pointer_cast(d_uv_selected.data()), thrust::raw_pointer_cast(d_opacity_selected.data()),
      thrust::raw_pointer_cast(pass_data.d_conic.data()), thrust::raw_pointer_cast(pass_data.d_precomputed_rgb.data()),
      bg_color, thrust::raw_pointer_cast(pass_data.d_sorted_gaussians.data()),
      thrust::raw_pointer_cast(pass_data.d_splat_start_end_idx_by_tile_idx.data()),
      thrust::raw_pointer_cast(pass_data.d_splats_per_pixel.data()),
      thrust::raw_pointer_cast(pass_data.d_weight_per_pixel.data()), thrust::raw_pointer_cast(d_grad_image.data()),
      width, height, thrust::raw_pointer_cast(cuda.gradients.d_grad_precompute_rgb.data()),
      thrust::raw_pointer_cast(cuda.gradients.d_grad_opacity.data()),
      thrust::raw_pointer_cast(cuda.gradients.d_grad_uv.data()),
      thrust::raw_pointer_cast(cuda.gradients.d_grad_conic.data()));

  precompute_spherical_harmonics_backward(thrust::raw_pointer_cast(d_xyz_c_selected.data()),
                                          thrust::raw_pointer_cast(cuda.gradients.d_grad_precompute_rgb.data()), l_max,
                                          pass_data.num_culled,
                                          thrust::raw_pointer_cast(cuda.gradients.d_grad_sh.data()),
                                          thrust::raw_pointer_cast(cuda.gradients.d_grad_rgb.data()));
  compute_conic_backward(
      thrust::raw_pointer_cast(pass_data.d_J.data()), thrust::raw_pointer_cast(pass_data.d_sigma.data()),
      thrust::raw_pointer_cast(cuda.camera.d_T.data()), thrust::raw_pointer_cast(cuda.gradients.d_grad_conic.data()),
      pass_data.num_culled, thrust::raw_pointer_cast(cuda.gradients.d_grad_J.data()),
      thrust::raw_pointer_cast(cuda.gradients.d_grad_sigma.data()));
  compute_projection_jacobian_backward(thrust::raw_pointer_cast(d_xyz_c_selected.data()),
                                       thrust::raw_pointer_cast(cuda.camera.d_K.data()),
                                       thrust::raw_pointer_cast(cuda.gradients.d_grad_J.data()), pass_data.num_culled,
                                       thrust::raw_pointer_cast(cuda.gradients.d_grad_xyz_c.data()));
  compute_sigma_backward(thrust::raw_pointer_cast(d_quaternion_selected.data()),
                         thrust::raw_pointer_cast(d_scale_selected.data()),
                         thrust::raw_pointer_cast(cuda.gradients.d_grad_sigma.data()), pass_data.num_culled,
                         thrust::raw_pointer_cast(cuda.gradients.d_grad_quaternion.data()),
                         thrust::raw_pointer_cast(cuda.gradients.d_grad_scale.data()));
  camera_intrinsic_projection_backward(thrust::raw_pointer_cast(d_xyz_c_selected.data()),
                                       thrust::raw_pointer_cast(cuda.camera.d_K.data()),
                                       thrust::raw_pointer_cast(cuda.gradients.d_grad_uv.data()), pass_data.num_culled,
                                       thrust::raw_pointer_cast(cuda.gradients.d_grad_xyz_c.data()));
  camera_extrinsic_projection_backward(
      thrust::raw_pointer_cast(d_xyz_selected.data()), thrust::raw_pointer_cast(cuda.camera.d_T.data()),
      thrust::raw_pointer_cast(cuda.gradients.d_grad_xyz_c.data()), pass_data.num_culled,
      thrust::raw_pointer_cast(cuda.gradients.d_grad_xyz.data()));

  return loss;
}

// A functor to compute the norm of a 2D gradient
struct PositionalGradientNorm {
  __host__ __device__ float operator()(const float2 &grad) const {
    const float u = grad.x;
    const float v = grad.y;
    return sqrtf(u * u + v * v);
  }
};

void TrainerImpl::optimizer_step(ForwardPassData pass_data) {
  auto d_xyz = compact_masked_array<3>(cuda.gaussians.d_xyz, pass_data.d_mask, pass_data.num_culled);
  auto d_rgb = compact_masked_array<3>(cuda.gaussians.d_rgb, pass_data.d_mask, pass_data.num_culled);
  auto d_op = compact_masked_array<1>(cuda.gaussians.d_opacity, pass_data.d_mask, pass_data.num_culled);
  auto d_scale = compact_masked_array<3>(cuda.gaussians.d_scale, pass_data.d_mask, pass_data.num_culled);
  auto d_quat = compact_masked_array<4>(cuda.gaussians.d_quaternion, pass_data.d_mask, pass_data.num_culled);

  auto d_m_xyz = compact_masked_array<3>(cuda.optimizer.m_grad_xyz, pass_data.d_mask, pass_data.num_culled);
  auto d_m_rgb = compact_masked_array<3>(cuda.optimizer.m_grad_rgb, pass_data.d_mask, pass_data.num_culled);
  auto d_m_op = compact_masked_array<1>(cuda.optimizer.m_grad_opacity, pass_data.d_mask, pass_data.num_culled);
  auto d_m_scale = compact_masked_array<3>(cuda.optimizer.m_grad_scale, pass_data.d_mask, pass_data.num_culled);
  auto d_m_quat = compact_masked_array<4>(cuda.optimizer.m_grad_quaternion, pass_data.d_mask, pass_data.num_culled);

  auto d_v_xyz = compact_masked_array<3>(cuda.optimizer.v_grad_xyz, pass_data.d_mask, pass_data.num_culled);
  auto d_v_rgb = compact_masked_array<3>(cuda.optimizer.v_grad_rgb, pass_data.d_mask, pass_data.num_culled);
  auto d_v_op = compact_masked_array<1>(cuda.optimizer.v_grad_opacity, pass_data.d_mask, pass_data.num_culled);
  auto d_v_scale = compact_masked_array<3>(cuda.optimizer.v_grad_scale, pass_data.d_mask, pass_data.num_culled);
  auto d_v_quat = compact_masked_array<4>(cuda.optimizer.v_grad_quaternion, pass_data.d_mask, pass_data.num_culled);

  const float bias1 = 1.0f - pow(B1, iter + 1);
  const float bias2 = 1.0f - pow(B2, iter + 1);

  const float xyz_decay_factor =
      pow((config.xyz_lr_multiplier_final / config.xyz_lr_multiplier_init), ((float)iter / (float)config.num_iters));
  adam_step(thrust::raw_pointer_cast(d_xyz.data()), thrust::raw_pointer_cast(cuda.gradients.d_grad_xyz.data()),
            thrust::raw_pointer_cast(d_m_xyz.data()), thrust::raw_pointer_cast(d_v_xyz.data()),
            scene_extent * config.base_lr * config.xyz_lr_multiplier_init * xyz_decay_factor, B1, B2, EPS, bias1, bias2,
            pass_data.num_culled, 3);
  adam_step(thrust::raw_pointer_cast(d_rgb.data()), thrust::raw_pointer_cast(cuda.gradients.d_grad_rgb.data()),
            thrust::raw_pointer_cast(d_m_rgb.data()), thrust::raw_pointer_cast(d_v_rgb.data()),
            config.base_lr * config.rgb_lr_multiplier, B1, B2, EPS, bias1, bias2, pass_data.num_culled, 3);
  adam_step(thrust::raw_pointer_cast(d_op.data()), thrust::raw_pointer_cast(cuda.gradients.d_grad_opacity.data()),
            thrust::raw_pointer_cast(d_m_op.data()), thrust::raw_pointer_cast(d_v_op.data()),
            config.base_lr * config.opacity_lr_multiplier, B1, B2, EPS, bias1, bias2, pass_data.num_culled, 1);
  adam_step(thrust::raw_pointer_cast(d_scale.data()), thrust::raw_pointer_cast(cuda.gradients.d_grad_scale.data()),
            thrust::raw_pointer_cast(d_m_scale.data()), thrust::raw_pointer_cast(d_v_scale.data()),
            config.base_lr * config.scale_lr_multiplier, B1, B2, EPS, bias1, bias2, pass_data.num_culled, 3);
  adam_step(thrust::raw_pointer_cast(d_quat.data()), thrust::raw_pointer_cast(cuda.gradients.d_grad_quaternion.data()),
            thrust::raw_pointer_cast(d_m_quat.data()), thrust::raw_pointer_cast(d_v_quat.data()),
            config.base_lr * config.quat_lr_multiplier, B1, B2, EPS, bias1, bias2, pass_data.num_culled, 4);

  scatter_masked_array<3>(d_m_xyz, pass_data.d_mask, cuda.optimizer.m_grad_xyz);
  scatter_masked_array<3>(d_m_rgb, pass_data.d_mask, cuda.optimizer.m_grad_rgb);
  scatter_masked_array<1>(d_m_op, pass_data.d_mask, cuda.optimizer.m_grad_opacity);
  scatter_masked_array<3>(d_m_scale, pass_data.d_mask, cuda.optimizer.m_grad_scale);
  scatter_masked_array<4>(d_m_quat, pass_data.d_mask, cuda.optimizer.m_grad_quaternion);

  scatter_masked_array<3>(d_v_xyz, pass_data.d_mask, cuda.optimizer.v_grad_xyz);
  scatter_masked_array<3>(d_v_rgb, pass_data.d_mask, cuda.optimizer.v_grad_rgb);
  scatter_masked_array<1>(d_v_op, pass_data.d_mask, cuda.optimizer.v_grad_opacity);
  scatter_masked_array<3>(d_v_scale, pass_data.d_mask, cuda.optimizer.v_grad_scale);
  scatter_masked_array<4>(d_v_quat, pass_data.d_mask, cuda.optimizer.v_grad_quaternion);

  scatter_masked_array<3>(d_xyz, pass_data.d_mask, cuda.gaussians.d_xyz);
  scatter_masked_array<3>(d_rgb, pass_data.d_mask, cuda.gaussians.d_rgb);
  scatter_masked_array<1>(d_op, pass_data.d_mask, cuda.gaussians.d_opacity);
  scatter_masked_array<3>(d_scale, pass_data.d_mask, cuda.gaussians.d_scale);
  scatter_masked_array<4>(d_quat, pass_data.d_mask, cuda.gaussians.d_quaternion);

  if (l_max > 0) {
    // SH logic would go here
    thrust::device_vector<float> d_sh;
    thrust::device_vector<float> d_m_sh;
    thrust::device_vector<float> d_v_sh;
    switch (l_max) {
    case 0:
      break;
    case 1:
      d_sh = compact_masked_array<9>(cuda.gaussians.d_sh, pass_data.d_mask, pass_data.num_culled);
      d_m_sh = compact_masked_array<9>(cuda.optimizer.m_grad_sh, pass_data.d_mask, pass_data.num_culled);
      d_v_sh = compact_masked_array<9>(cuda.optimizer.v_grad_sh, pass_data.d_mask, pass_data.num_culled);

      adam_step(thrust::raw_pointer_cast(d_sh.data()), thrust::raw_pointer_cast(cuda.gradients.d_grad_sh.data()),
                thrust::raw_pointer_cast(d_m_sh.data()), thrust::raw_pointer_cast(d_v_sh.data()),
                config.base_lr * config.sh_lr_multiplier, B1, B2, EPS, bias1, bias2, pass_data.num_culled, 9);

      scatter_masked_array<9>(d_m_sh, pass_data.d_mask, cuda.optimizer.m_grad_sh);
      scatter_masked_array<9>(d_v_sh, pass_data.d_mask, cuda.optimizer.v_grad_sh);
      scatter_masked_array<9>(d_sh, pass_data.d_mask, cuda.gaussians.d_sh);
      break;
    case 2:
      d_sh = compact_masked_array<24>(cuda.gaussians.d_sh, pass_data.d_mask, pass_data.num_culled);
      d_m_sh = compact_masked_array<24>(cuda.optimizer.m_grad_sh, pass_data.d_mask, pass_data.num_culled);
      d_v_sh = compact_masked_array<24>(cuda.optimizer.v_grad_sh, pass_data.d_mask, pass_data.num_culled);
      adam_step(thrust::raw_pointer_cast(d_sh.data()), thrust::raw_pointer_cast(cuda.gradients.d_grad_sh.data()),
                thrust::raw_pointer_cast(d_m_sh.data()), thrust::raw_pointer_cast(d_v_sh.data()),
                config.base_lr * config.sh_lr_multiplier, B1, B2, EPS, bias1, bias2, pass_data.num_culled, 24);

      scatter_masked_array<24>(d_m_sh, pass_data.d_mask, cuda.optimizer.m_grad_sh);
      scatter_masked_array<24>(d_v_sh, pass_data.d_mask, cuda.optimizer.v_grad_sh);
      scatter_masked_array<24>(d_sh, pass_data.d_mask, cuda.gaussians.d_sh);
      break;
    case 3:
      d_sh = compact_masked_array<45>(cuda.gaussians.d_sh, pass_data.d_mask, pass_data.num_culled);
      d_m_sh = compact_masked_array<45>(cuda.optimizer.m_grad_sh, pass_data.d_mask, pass_data.num_culled);
      d_v_sh = compact_masked_array<45>(cuda.optimizer.v_grad_sh, pass_data.d_mask, pass_data.num_culled);
      adam_step(thrust::raw_pointer_cast(d_sh.data()), thrust::raw_pointer_cast(cuda.gradients.d_grad_sh.data()),
                thrust::raw_pointer_cast(d_m_sh.data()), thrust::raw_pointer_cast(d_v_sh.data()),
                config.base_lr * config.sh_lr_multiplier, B1, B2, EPS, bias1, bias2, pass_data.num_culled, 45);
      scatter_masked_array<45>(d_m_sh, pass_data.d_mask, cuda.optimizer.m_grad_sh);
      scatter_masked_array<45>(d_v_sh, pass_data.d_mask, cuda.optimizer.v_grad_sh);
      scatter_masked_array<45>(d_sh, pass_data.d_mask, cuda.gaussians.d_sh);
      break;
    default:
      fprintf(stderr, "Error SH band is invalid\n");
      exit(EXIT_FAILURE);
    }
  }

  // Update gradient accumulators after step
  // Compact
  auto d_uv_accum_compact =
      compact_masked_array<1>(cuda.accumulators.d_uv_grad_accum, pass_data.d_mask, pass_data.num_culled);
  auto d_accum_dur_compact =
      compact_masked_array<1>(cuda.accumulators.d_grad_accum_dur, pass_data.d_mask, pass_data.num_culled);
  // Add
  thrust::device_vector<float> d_uv_grad_norms(pass_data.num_culled);
  thrust::transform(reinterpret_cast<float2 *>(thrust::raw_pointer_cast(cuda.gradients.d_grad_uv.data())),
                    reinterpret_cast<float2 *>(thrust::raw_pointer_cast(cuda.gradients.d_grad_uv.data())) +
                        pass_data.num_culled,
                    d_uv_grad_norms.begin(), PositionalGradientNorm());
  thrust::transform(d_uv_accum_compact.begin(), d_uv_accum_compact.end(), d_uv_grad_norms.begin(),
                    d_uv_accum_compact.begin(), thrust::plus<float>());

  thrust::transform(d_accum_dur_compact.begin(), d_accum_dur_compact.end(), thrust::make_constant_iterator(1),
                    d_accum_dur_compact.begin(), thrust::plus<int>());

  // Scatter
  scatter_masked_array<1>(d_uv_accum_compact, pass_data.d_mask, cuda.accumulators.d_uv_grad_accum);
  scatter_masked_array<1>(d_accum_dur_compact, pass_data.d_mask, cuda.accumulators.d_grad_accum_dur);
}

void TrainerImpl::train() {
  // Call the Impl member function
  reset_grad_accum();

  // Copy Gaussian data from host (member 'gaussians') to device (member 'cuda.gaussians')
  try {
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

  // Assume max resolution for Mip-NeRF 360 dataset (50 megapixels)
  init_pinned_memory(5187 * 3361);

  shutdown_requested = false;

  // --- Prologue: Load and Copy first image ---
  {
    // 1. Request Load Image 0
    std::unique_lock<std::mutex> lock(load_mutex);

    // Initialize RNG
    std::random_device rd;
    rng = std::mt19937(rd());
    std::uniform_int_distribution<int> dist(0, train_images.size() - 1);

    int first_idx = dist(rng);
    next_image_index = first_idx;
    buffer_image_indices[0] = first_idx;

    request_load[0] = true;
    loader_thread = std::thread(&TrainerImpl::image_loader_loop, this);

    // Wait for Image 0 to be ready in Host Buffer 0
    load_cv.wait(lock, [this] { return buffer_ready[0]; });
    buffer_ready[0] = false; // Consume
    lock.unlock();

    Image curr_image = train_images[first_idx];
    Camera curr_camera = cameras[curr_image.camera_id];
    int width = curr_camera.width;
    int height = curr_camera.height;

    if (d_gt_image[0].size() != width * height * 3) {
      d_gt_image[0].resize(width * height * 3);
    }

    // 2. Launch Copy Image 0 -> Device Buffer 0
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_gt_image[0].data()), h_pinned_image_buffer[0],
                    width * height * 3 * sizeof(float), cudaMemcpyHostToDevice, transfer_stream);
    cudaEventRecord(copy_finished_events[0], transfer_stream);

    // 3. Request Load Image 1 -> Host Buffer 1 (Async)
    lock.lock();
    if (train_images.size() > 1) {
      std::uniform_int_distribution<int> dist(0, train_images.size() - 1);
      int next_idx = dist(rng);
      next_image_index = next_idx;
      buffer_image_indices[1] = next_idx;
      request_load[1] = true;
      load_cv.notify_all();
    }
    lock.unlock();
  }

  // Calculate scene extent for adaptive density
  scene_extent = 1.1f * computeMaxDiagonal(images);

  ProgressBar progressBar(config.num_iters);

  // TRAINING LOOP
  while (iter < config.num_iters) {
    int curr_buf_idx = iter % 2;
    int next_buf_idx = (iter + 1) % 2;

    Image curr_image = train_images[buffer_image_indices[curr_buf_idx]];
    Camera curr_camera = cameras[curr_image.camera_id];

    // 1. Wait for GT Image Transfer to complete on GPU
    // This ensures that the compute stream doesn't start using d_gt_image[curr] until copy is done.
    cudaStreamWaitEvent(0, copy_finished_events[curr_buf_idx], 0);

    // 2. Submit Compute Work (Async)
    ForwardPassData pass_data;
    zero_grads();

    // Prepare and copy camera parameters to device (member 'cuda.camera')
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
      thrust::copy(h_K, h_K + 9, cuda.camera.d_K.begin());
      thrust::copy(h_T, h_T + 12, cuda.camera.d_T.begin());
    } catch (const std::exception &e) {
      fprintf(stderr, "Error copying camera data to device: %s\\n", e.what());
      exit(EXIT_FAILURE);
    }

    float bg_color = 0.0f;
    if (config.use_background)
      bg_color = (iter % 255) / 255.0f;

    // Call Impl member function
    if (iter % config.add_sh_band_interval == 0 && iter >= config.add_sh_band_interval)
      add_sh_band();

    // --- FORWARD PASS via RASTERIZE MODULE ---
    rasterize_image(num_gaussians, curr_camera, config, cuda.camera, cuda.gaussians, pass_data, bg_color, l_max);

    if (pass_data.num_culled == 0) {
      std::cerr << "WARNING Image " << curr_image.id << " has no Gaussians in view" << std::endl;
    } else {
      // --- BACKWARD PASS ---
      float loss = backward_pass(curr_image, curr_camera, pass_data, bg_color, d_gt_image[curr_buf_idx]);

      // --- OPTIMIZER STEP ---
      optimizer_step(pass_data);

      // Log status
      progressBar.update(iter, loss, num_gaussians);
    }
    // --- SAVE RENDERED IMAGE ---
    if (iter % config.print_interval == 0) {
      const int width = (int)curr_camera.width;
      const int height = (int)curr_camera.height;
      std::vector<float> h_image_buffer(width * height * 3);
      thrust::copy(pass_data.d_image_buffer.begin(), pass_data.d_image_buffer.end(), h_image_buffer.begin());

      cv::Mat rendered_image(height, width, CV_32FC3, h_image_buffer.data());
      cv::Mat rendered_image_8u;
      rendered_image.convertTo(rendered_image_8u, CV_8UC3, 255.0);
      cv::cvtColor(rendered_image_8u, rendered_image_8u, cv::COLOR_RGB2BGR);

      std::string filename = "rendered_image_" + std::to_string(iter) + ".png";
      cv::imwrite(filename, rendered_image_8u);
    }

    // --- EVALUATION ---
    if (iter % 3000 == 0) {
      evaluate();
    }

    // --- ADAPTIVE DENSITY ---
    if (iter > config.adaptive_control_start && iter % config.adaptive_control_interval == 0 &&
        iter < config.adaptive_control_end) {
      adaptive_density_step();
      reset_grad_accum();
    }

    if (iter > config.reset_opacity_start && iter % config.reset_opacity_interval == 0 &&
        iter < config.reset_opacity_end) {
      reset_opacity();
      reset_grad_accum();
    }

    // 3. Launch Transfer for Next Image (iter + 1)
    // We need to check if Host Buffer for next image is ready.
    {
      std::unique_lock<std::mutex> lock(load_mutex);
      // Wait for loader to finish filling the buffer
      load_cv.wait(lock, [this, next_buf_idx] { return buffer_ready[next_buf_idx]; });
      buffer_ready[next_buf_idx] = false; // Consumed
    }

    // Resize if needed
    Image next_image = train_images[(iter + 1) % train_images.size()];
    Camera next_camera = cameras[next_image.camera_id];
    int next_width = next_camera.width;
    int next_height = next_camera.height;

    if (d_gt_image[next_buf_idx].size() != next_width * next_height * 3) {
      d_gt_image[next_buf_idx].resize(next_width * next_height * 3);
    }

    cudaMemcpyAsync(thrust::raw_pointer_cast(d_gt_image[next_buf_idx].data()), h_pinned_image_buffer[next_buf_idx],
                    next_width * next_height * 3 * sizeof(float), cudaMemcpyHostToDevice, transfer_stream);
    cudaEventRecord(copy_finished_events[next_buf_idx], transfer_stream);

    // 4. Trigger Load for Image (iter + 2) into 'curr_buf_idx'
    // We must ensure that the PREVIOUS copy from 'curr_buf_idx' (which was for image 'iter') is done.
    // That copy was recorded in 'copy_finished_events[curr_buf_idx]'.
    cudaEventSynchronize(copy_finished_events[curr_buf_idx]);

    {
      std::unique_lock<std::mutex> lock(load_mutex);
      std::uniform_int_distribution<int> dist(0, train_images.size() - 1);
      int next_idx = dist(rng);

      next_image_index = next_idx;
      buffer_image_indices[curr_buf_idx] = next_idx; // Update for the next time this buffer is used

      request_load[curr_buf_idx] = true;
      load_cv.notify_all();
    }

    iter++;
  }
  // Cleanup
  {
    std::lock_guard<std::mutex> lock(load_mutex);
    shutdown_requested = true;
    load_cv.notify_all();
  }
  if (loader_thread.joinable())
    loader_thread.join();
  free_pinned_memory();
  progressBar.finish();
}

void TrainerImpl::evaluate() {
  if (test_images.empty())
    return;

  std::cout << "\n[ITER " << iter << "] Evaluating on " << test_images.size() << " test images..." << std::endl;

  float total_psnr = 0.0f;

  // Use a separate device buffer for test image to avoid conflict with double buffering
  thrust::device_vector<float> d_test_image;

  for (const auto &img : test_images) {
    // Load image synchronously
    cv::Mat bgr_image = cv::imread(img.name, cv::IMREAD_COLOR);
    if (bgr_image.empty()) {
      std::cerr << "Failed to load test image: " << img.name << std::endl;
      continue;
    }

    Camera cam = cameras[img.camera_id];

    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    int width = cam.width;
    int height = cam.height;

    // Resize device buffer if needed
    if (d_test_image.size() != width * height * 3) {
      d_test_image.resize(width * height * 3);
    }

    // Copy to device
    cudaMemcpy(thrust::raw_pointer_cast(d_test_image.data()), float_image.ptr<float>(0),
               width * height * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Prepare camera data
    float h_K[9] = {(float)cam.params[0],
                    0.f,
                    (float)cam.params[2],
                    0.f,
                    (float)cam.params[1],
                    (float)cam.params[3],
                    0.f,
                    0.f,
                    1.f};
    Eigen::Matrix3d rot_mat_d = img.QvecToRotMat();
    Eigen::Vector3d t_vec_d = img.tvec;
    float h_T[12];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j)
        h_T[i * 4 + j] = (float)rot_mat_d(i, j);
      h_T[i * 4 + 3] = (float)t_vec_d(i);
    }
    thrust::copy(h_K, h_K + 9, cuda.camera.d_K.begin());
    thrust::copy(h_T, h_T + 12, cuda.camera.d_T.begin());

    // Render
    ForwardPassData pass_data;
    float bg_color = 0.0f; // Black background for eval

    rasterize_image(num_gaussians, cam, config, cuda.camera, cuda.gaussians, pass_data, bg_color, l_max);

    // Compute PSNR
    float psnr = compute_psnr(thrust::raw_pointer_cast(pass_data.d_image_buffer.data()),
                              thrust::raw_pointer_cast(d_test_image.data()), height, width);
    total_psnr += psnr;
  }

  float avg_psnr = total_psnr / test_images.size();
  std::cout << "[ITER " << iter << "] Eval PSNR: " << avg_psnr << std::endl;
}

// --- Implementation of Public Trainer Methods ---
// These methods simply delegate to the pImpl object.

Trainer::Trainer()
    : pImpl(std::make_unique<TrainerImpl>(ConfigParameters{}, Gaussians{}, std::unordered_map<int, Image>{},
                                          std::unordered_map<int, Camera>{})) {}

Trainer::Trainer(ConfigParameters config, Gaussians gaussians, std::unordered_map<int, Image> images,
                 std::unordered_map<int, Camera> cameras)
    : pImpl(std::make_unique<TrainerImpl>(std::move(config), std::move(gaussians), std::move(images),
                                          std::move(cameras))) {}

// Define the destructor and move operations.
Trainer::~Trainer() = default;
Trainer::Trainer(Trainer &&) noexcept = default;
Trainer &Trainer::operator=(Trainer &&) noexcept = default;

// --- Public API Delegation ---

void Trainer::test_train_split() { pImpl->test_train_split(); }

void Trainer::train() { pImpl->train(); }
