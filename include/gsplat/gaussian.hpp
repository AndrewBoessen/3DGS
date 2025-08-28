// gaussian.hpp

#pragma once

#include <dataloader/colmap.hpp>
#include <optional>
#include <vector>

#include <Eigen/Dense>

/**
 * @class Gaussians
 * @brief Contains all mutable parameters for a set of 3D Gaussian splats.
 */
class Gaussians {
public:
  /// @brief Positions (centers) of the Gaussians [x, y, z].
  std::vector<Eigen::Vector3f> xyz;

  /// @brief Colors of the Gaussians [r, g, b], normalized to [0, 1].
  std::vector<Eigen::Vector3f> rgb;

  /// @brief Opacity values for each Gaussian, typically in [0, 1].
  std::vector<float> opacity;

  /// @brief Scaling factors for each Gaussian [sx, sy, sz].
  std::vector<Eigen::Vector3f> scale;

  /// @brief Rotations for each Gaussian as quaternions [w, x, y, z].
  std::vector<Eigen::Quaternionf> quaternion;

  /// @brief Optional Spherical Harmonics (SH) coefficients.
  /// Each Eigen::VectorXf contains all coefficients for a single Gaussian.
  std::optional<std::vector<Eigen::VectorXf>> sh;

  /**
   * @brief Default constructor. Creates an empty Gaussians object.
   */
  Gaussians() = default;

  /**
   * @brief Constructs a Gaussians object by moving data from the provided vectors.
   */
  Gaussians(std::vector<Eigen::Vector3f> &&xyz_in, std::vector<Eigen::Vector3f> &&rgb_in,
            std::vector<float> &&opacity_in, std::vector<Eigen::Vector3f> &&scale_in,
            std::vector<Eigen::Quaternionf> &&quaternion_in,
            std::optional<std::vector<Eigen::VectorXf>> &&sh_in = std::nullopt)
      : xyz(std::move(xyz_in)), rgb(std::move(rgb_in)), opacity(std::move(opacity_in)), scale(std::move(scale_in)),
        quaternion(std::move(quaternion_in)), sh(std::move(sh_in)) {}

  /**
   * @brief Returns the number of Gaussian primitives stored.
   */
  size_t size() const { return xyz.size(); }

  /**
   * @brief Initialize Gaussians on points in Point3D.
   * @param points Set of points in point cloud.
   * @return Gaussians with default parameters.
   */
  static Gaussians Initialize(const std::unordered_map<uint64_t, Point3D> &points);

  /**
   * @brief Appends new Gaussians from another Gaussians object.
   * @param other The Gaussians object to append.
   */
  void append(const Gaussians &other);

  /**
   * @brief Filters the current Gaussians based on a boolean mask.
   * @param mask A boolean vector of the same size as the number of Gaussians.
   */
  void filter(const std::vector<bool> &mask);
};
