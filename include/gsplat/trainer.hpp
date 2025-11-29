// trainer.hpp

#pragma once

#include "dataloader/colmap.hpp"
#include "gsplat/gaussian.hpp"
#include "gsplat/utils.hpp"
#include <memory>

// Forward declaration of the implementation class.
// The full definition will be hidden in the .cu file.
class TrainerImpl;

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
  Trainer();

  /**
   * @brief Constructs a Trainer with initial configuration and scene data.
   * @param config The configuration parameters for the training process.
   * @param gaussians The initial set of 3D Gaussians to be trained.
   * @param images A map of image data keyed by ID.
   * @param cameras A map of camera parameters corresponding to the images, keyed by ID.
   */
  Trainer(ConfigParameters config, Gaussians gaussians, std::unordered_map<int, Image> images,
          std::unordered_map<int, Camera> cameras);

  /**
   * @brief Destructor.
   * @note Must be defined in the .cu file where TrainerImpl is a complete type.
   */
  ~Trainer();

  // --- Manage resource ownership with move semantics ---
  Trainer(Trainer &&) noexcept;
  Trainer &operator=(Trainer &&) noexcept;

  // --- Disable copy semantics ---
  Trainer(const Trainer &) = delete;
  Trainer &operator=(const Trainer &) = delete;

  /**
   * @brief Splits the full image dataset into training and testing sets.
   * @note This method uses a deterministic "every N-th image" strategy based on the
   * `test_split_ratio` in the configuration. Images are sorted by name before
   * splitting to ensure reproducibility across runs.
   */
  void test_train_split();

  /**
   * @brief Starts and manages the main training loop.
   * @note This function orchestrates the entire training process, including forward
   * and backward passes, optimizer steps, and calls to adaptive density control
   * functions at specified intervals.
   */
  void train();

  /**
   * @brief Save trained Gaussians to PLY file
   * @param[in] filename Output filename
   */
  void save_to_ply(const std::string filename);

private:
  /// @brief Pointer to the private implementation.
  std::unique_ptr<TrainerImpl> pImpl;
};
