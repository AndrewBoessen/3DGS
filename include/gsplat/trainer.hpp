// trainer.hpp

#pragma once

#include "dataloader/colmap.hpp"
#include "gsplat/cuda_data.hpp"
#include "gsplat/gaussian.hpp"
#include "gsplat/utils.hpp"
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

/**
 * @brief Manages the entire training process for 3D Gaussian Splatting.
 *
 * This class orchestrates the training loop, including data handling,
 * optimization of Gaussian parameters, and adaptive density control (splitting
 * and cloning Gaussians) to refine the 3D scene representation.
 */
class Trainer {
public:
  /**
   * @brief Default constructor for the Trainer class.
   */
  Trainer() = default;

  /**
   * @brief Constructs a Trainer with initial configuration and scene data.
   * @param config The configuration parameters for the training process.
   * @param gaussians The initial set of 3D Gaussians to be trained.
   * @param images A map of image data keyed by ID.
   * @param cameras A map of camera parameters corresponding to the images, keyed by ID.
   */
  Trainer(ConfigParameters config, Gaussians gaussians, std::unordered_map<int, Image> images,
          std::unordered_map<int, Camera> cameras)
      : config(std::move(config)), gaussians(std::move(gaussians)), images(std::move(images)),
        cameras(std::move(cameras)) {}

  /**
   * @breif Resets the gradient accumulation for xyz and uv.
   * @note This will set all values for Gaussians to zero. This must be run
   * only after the adaptive density in run and `gaussians` is resized.
   */
  void reset_grad_accum();

  /**
   * @brief Splits the full image dataset into training and testing sets.
   * @note This method uses a deterministic "every N-th image" strategy based on the
   * `test_split_ratio` in the configuration. Images are sorted by name before
   * splitting to ensure reproducibility across runs.
   */
  void test_train_split();

  /**
   * @brief Resets the opacity of all Gaussians to a predefined value.
   * @note This is typically done at the beginning of training to help prune
   * Gaussians that do not contribute significantly to the rendered images. The
   * value is transformed into logit space before being assigned.
   */
  void reset_opacity();

  /**
   * @brief Increases the spherical harmonics (SH) degree for all Gaussians.
   * @note This function progressively increases the complexity of the Gaussian
   * color representation, allowing for more detailed and view-dependent effects
   * as training progresses. It upgrades the SH bands up to the maximum level
   * specified in the configuration.
   */
  void add_sh_band();

  /**
   * @brief Manages the adaptive densification of Gaussians during training.
   * @note This function is intended to be called periodically. It identifies
   * Gaussians that need to be split (to represent finer details) or cloned (to
   * fill in under-reconstructed areas) based on their gradients.
   */
  void adaptive_density();

  /**
   * @brief Starts and manages the main training loop.
   * @note This function orchestrates the entire training process, including forward
   * and backward passes, optimizer steps, and calls to adaptive density control
   * functions at specified intervals.
   */
  void train();

private:
  /// @brief Configuration parameters for the training session.
  ConfigParameters config;
  /// @brief The collection of 3D Gaussians being optimized.
  Gaussians gaussians;
  /// @brief A map of all images in the dataset, keyed by image ID.
  std::unordered_map<int, Image> images;
  /// @brief A map of all cameras in the dataset, keyed by camera ID.
  std::unordered_map<int, Camera> cameras;

  /// @brief A vector of images designated for the testing set.
  std::vector<Image> test_images;
  /// @brief A vector of images designated for the training set.
  std::vector<Image> train_images;

  /// @brief A vector to hold sum of gradients of image view.
  std::vector<Eigen::Vector2f> uv_grad_accum;

  /// @brief A vector to hold sum world corrdinates.
  std::vector<Eigen::Vector3f> xyz_grad_accum;

  /// @brief The duration of gradient accumulation.
  std::vector<int> grad_accum_dur;

  /// @brief The current training iteration count.
  int iter = 0;

  /**
   * @brief Splits specified Gaussians into multiple, smaller Gaussians.
   * @param split_mask A boolean vector where `true` indicates the Gaussian at
   * the corresponding index should be split.
   * @note This method replaces large Gaussians with several smaller ones sampled
   * from their distribution. This helps to model finer details in the scene.
   * The scales of the new Gaussians are reduced by a configurable factor.
   */
  void split_gaussians(const std::vector<bool> &split_mask);

  /**
   * @brief Clones specified Gaussians to increase density in under-reconstructed areas.
   * @param clone_mask A boolean vector where `true` indicates the Gaussian at
   * the corresponding index should be cloned.
   * @param xyz_grad_avg A vector with average xyz gradients. This is used to set the
   * position of cloned Gaussians.
   * @note This method duplicates Gaussians in place, effectively increasing their
   * influence in regions that require more geometric detail.
   */
  void clone_gaussians(const std::vector<bool> &clone_mask, const std::vector<Eigen::Vector3f> &xyz_grad_avg);

  /**
   * @brief Free temporary memory buffers for training iteration
   * @param[in] pass_data Temporary buffer data
   */
  void cleanup_iteration_buffers(ForwardPassData &pass_data);

  /**
   * @brief Compute gradients from forward pass
   *
   * @param[in] curr_image Rendered image data
   * @param[in] curr_camera Current camera parameters
   * @param[in] cuda Device data to store gradients in
   * @param[in] pass_data Forward pass temporary buffers
   * @param[in] streams CUDA streams to use
   * @return Loss value
   */
  float backward_pass(const Image &curr_image, const Camera &curr_camera, CudaDataManager &cuda,
                      ForwardPassData &pass_data, const std::vector<cudaStream_t> &streams);
};
