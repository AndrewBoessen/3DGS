// optimize.hpp

#pragma once

#include "gsplat/gaussian.hpp"
#include <Eigen/Dense>
#include <vector>

/**
 * @struct Gradients
 * @brief Holds the gradients for each optimizable parameter in the Gaussians class.
 *
 * This structure mirrors the layout of the Gaussians class for clarity and ease of use.
 */
struct Gradients {
  std::vector<Eigen::Vector3f> xyz;
  std::vector<Eigen::Vector3f> rgb;
  std::vector<float> opacity;
  std::vector<Eigen::Vector3f> scale;
  /// @brief Gradients for quaternions are 3D vectors in the tangent space.
  std::vector<Eigen::Vector3f> quaternion;
  std::optional<std::vector<Eigen::VectorXf>> sh;
};

/**
 * @class AdamOptimizer
 * @brief Implements the Adam optimization algorithm for 3D Gaussian Splatting models.
 *
 * This optimizer manages the first and second moment estimates (m and v) for all
 * Gaussian parameters and includes functionality to dynamically adjust its state
 * when Gaussians are added or removed during training.
 */
class AdamOptimizer {
public:
  /**
   * @brief Constructs an AdamOptimizer.
   * @param lr The learning rate.
   * @param beta1 Exponential decay rate for the first moment estimates.
   * @param beta2 Exponential decay rate for the second moment estimates.
   * @param epsilon A small constant for numerical stability.
   */
  AdamOptimizer(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

  /**
   * @brief Performs a single optimization step, updating the Gaussian parameters.
   * @param params A reference to the Gaussians object to be updated.
   * @param grads A const reference to the Gradients object containing the derivatives.
   */
  void step(Gaussians &params, const Gradients &grads);

  /**
   * @brief Filters the optimizer's state (m and v vectors) using a boolean mask.
   *
   * This method **must** be called after filtering the main Gaussians object to
   * ensure the optimizer's state remains synchronized with the parameters.
   * @param mask A boolean vector where `true` indicates an element to keep.
   */
  void filter_states(const std::vector<bool> &mask);

  /**
   * @brief Appends zero-initialized states for a number of new Gaussians.
   *
   * This method **must** be called after new Gaussians are appended (densified)
   * to initialize their corresponding optimizer states.
   * @param n The number of new Gaussians that were added.
   */
  void append_states(size_t n);

private:
  // Hyperparameters
  float lr_;
  float beta1_;
  float beta2_;
  float epsilon_;
  int t_{0}; // Timestep counter, starts at 0

  // Struct to hold the first and second moment estimates for all parameters.
  struct OptimizerState {
    std::vector<Eigen::Vector3f> m_xyz, v_xyz;
    std::vector<Eigen::Vector3f> m_rgb, v_rgb;
    std::vector<float> m_opacity, v_opacity;
    std::vector<Eigen::Vector3f> m_scale, v_scale;
    std::vector<Eigen::Vector3f> m_quaternion, v_quaternion;
    std::optional<std::vector<Eigen::VectorXf>> m_sh, v_sh;
  } state_;

  /**
   * @brief Initializes state vectors with the correct size and zeros on the first step.
   * @param params The initial set of Gaussians.
   */
  void initialize_states_if_needed(const Gaussians &params);
};
