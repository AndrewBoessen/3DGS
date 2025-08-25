// colmap.hpp

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

struct CameraModel {
  int model_id;
  std::string model_name;
  int num_params;
};

struct Camera {
  int id;
  std::string model;
  uint64_t width;
  uint64_t height;
  std::vector<double> params;
};

struct Image {
  int id;
  Eigen::Vector4d qvec; // Quaternion as (w, x, y, z)
  Eigen::Vector3d tvec;
  int camera_id;
  std::string name;
  std::vector<Eigen::Vector2d> xys;
  std::vector<int64_t> point3D_ids;

  // Member function to convert quaternion to rotation matrix.
  [[nodiscard]] Eigen::Matrix3d QvecToRotMat() const;
};

struct Point3D {
  uint64_t id;
  Eigen::Vector3d xyz;
  std::array<uint8_t, 3> rgb;
  double error;
  std::vector<int> image_ids;
  std::vector<int> point2D_idxs;
};

// Provides access to the global map of known COLMAP camera models.
const std::unordered_map<int, CameraModel> &GetCameraModels();

/**
 * @brief Reads camera data from a COLMAP cameras.bin file.
 * @param path The path to the cameras.bin file.
 * @return An optional map from camera ID to Camera struct.
 * Returns std::nullopt if the file cannot be opened or read.
 */
std::optional<std::unordered_map<int, Camera>> ReadCamerasBinary(const std::filesystem::path &path);

/**
 * @brief Reads image data from a COLMAP images.bin file.
 * @param path The path to the images.bin file.
 * @return An optional map from image ID to Image struct.
 * Returns std::nullopt if the file cannot be opened or read.
 */
std::optional<std::unordered_map<int, Image>> ReadImagesBinary(const std::filesystem::path &path);

/**
 * @brief Reads 3D point data from a COLMAP points3D.bin file.
 * @param path The path to the points3D.bin file.
 * @return An optional map from point ID to Point3D struct.
 * Returns std::nullopt if the file cannot be opened or read.
 */
std::optional<std::unordered_map<uint64_t, Point3D>> ReadPoints3DBinary(const std::filesystem::path &path);

} // namespace colmap
