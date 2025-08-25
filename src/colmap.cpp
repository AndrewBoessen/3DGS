// colmap.cpp

#include "dataloader/colmap.hpp"

#include <fstream>

namespace colmap {

namespace {
// Helper template to read a data type from a binary stream.
template <typename T> bool ReadBinary(std::ifstream &stream, T &value) {
  return static_cast<bool>(stream.read(reinterpret_cast<char *>(&value), sizeof(T)));
}
} // namespace

const std::unordered_map<int, CameraModel> &GetCameraModels() {
  static const std::unordered_map<int, CameraModel> CAMERA_MODELS = {
      {0, {0, "SIMPLE_PINHOLE", 3}},
      {1, {1, "PINHOLE", 4}},
      {2, {2, "SIMPLE_RADIAL", 4}},
      {3, {3, "RADIAL", 5}},
      {4, {4, "OPENCV", 8}},
      {5, {5, "OPENCV_FISHEYE", 8}},
      {6, {6, "FULL_OPENCV", 12}},
      {7, {7, "FOV", 5}},
      {8, {8, "SIMPLE_RADIAL_FISHEYE", 4}},
      {9, {9, "RADIAL_FISHEYE", 5}},
      {10, {10, "THIN_PRISM_FISHEYE", 12}},
  };
  return CAMERA_MODELS;
}

Eigen::Matrix3d Image::QvecToRotMat() const {
  Eigen::Quaterniond q(qvec(0), qvec(1), qvec(2), qvec(3));
  return q.toRotationMatrix();
}

std::optional<std::unordered_map<int, Camera>> ReadCamerasBinary(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Error: Could not open file " << path << std::endl;
    return std::nullopt;
  }

  uint64_t num_cameras;
  if (!ReadBinary(file, num_cameras)) {
    std::cerr << "Error: Failed to read number of cameras." << std::endl;
    return std::nullopt;
  }

  std::unordered_map<int, Camera> cameras;
  cameras.reserve(num_cameras);

  const auto &camera_models = GetCameraModels();

  for (uint64_t i = 0; i < num_cameras; ++i) {
    Camera cam;
    int model_id;
    if (!ReadBinary(file, cam.id) || !ReadBinary(file, model_id) || !ReadBinary(file, cam.width) ||
        !ReadBinary(file, cam.height)) {
      std::cerr << "Error: Failed to read camera properties." << std::endl;
      return std::nullopt;
    }

    auto it = camera_models.find(model_id);
    if (it == camera_models.end()) {
      std::cerr << "Error: Unknown camera model ID: " << model_id << std::endl;
      return std::nullopt;
    }
    cam.model = it->second.model_name;

    cam.params.resize(it->second.num_params);
    if (!file.read(reinterpret_cast<char *>(cam.params.data()), it->second.num_params * sizeof(double))) {
      std::cerr << "Error: Failed to read camera parameters." << std::endl;
      return std::nullopt;
    }
    cameras[cam.id] = std::move(cam);
  }
  return cameras;
}

std::optional<std::unordered_map<int, Image>> ReadImagesBinary(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Error: Could not open file " << path << std::endl;
    return std::nullopt;
  }

  uint64_t num_images;
  if (!ReadBinary(file, num_images)) {
    std::cerr << "Error: Failed to read number of images." << std::endl;
    return std::nullopt;
  }

  std::unordered_map<int, Image> images;
  images.reserve(num_images);

  for (uint64_t i = 0; i < num_images; ++i) {
    Image img;

    if (!ReadBinary(file, img.id) || !ReadBinary(file, img.qvec(0)) || !ReadBinary(file, img.qvec(1)) ||
        !ReadBinary(file, img.qvec(2)) || !ReadBinary(file, img.qvec(3)) || !ReadBinary(file, img.tvec(0)) ||
        !ReadBinary(file, img.tvec(1)) || !ReadBinary(file, img.tvec(2)) || !ReadBinary(file, img.camera_id)) {
      std::cerr << "Error: Failed to read image properties." << std::endl;
      return std::nullopt;
    }

    char name_char;
    while (ReadBinary(file, name_char) && name_char != '\0') {
      img.name += name_char;
    }

    uint64_t num_points2D;
    if (!ReadBinary(file, num_points2D)) {
      std::cerr << "Error: Failed to read number of 2D points." << std::endl;
      return std::nullopt;
    }

    img.xys.resize(num_points2D);
    img.point3D_ids.resize(num_points2D);

    for (uint64_t j = 0; j < num_points2D; ++j) {
      if (!ReadBinary(file, img.xys[j](0)) || !ReadBinary(file, img.xys[j](1)) ||
          !ReadBinary(file, img.point3D_ids[j])) {
        std::cerr << "Error: Failed to read 2D point data." << std::endl;
        return std::nullopt;
      }
    }
    images[img.id] = std::move(img);
  }
  return images;
}

std::optional<std::unordered_map<uint64_t, Point3D>> ReadPoints3DBinary(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Error: Could not open file " << path << std::endl;
    return std::nullopt;
  }

  uint64_t num_points;
  if (!ReadBinary(file, num_points)) {
    std::cerr << "Error: Failed to read number of 3D points." << std::endl;
    return std::nullopt;
  }

  std::unordered_map<uint64_t, Point3D> points3D;
  points3D.reserve(num_points);

  for (uint64_t i = 0; i < num_points; ++i) {
    Point3D p3d;

    if (!ReadBinary(file, p3d.id) || !ReadBinary(file, p3d.xyz(0)) || !ReadBinary(file, p3d.xyz(1)) ||
        !ReadBinary(file, p3d.xyz(2)) || !ReadBinary(file, p3d.rgb[0]) || !ReadBinary(file, p3d.rgb[1]) ||
        !ReadBinary(file, p3d.rgb[2]) || !ReadBinary(file, p3d.error)) {
      std::cerr << "Error: Failed to read 3D point properties." << std::endl;
      return std::nullopt;
    }

    uint64_t track_length;
    if (!ReadBinary(file, track_length)) {
      std::cerr << "Error: Failed to read track length." << std::endl;
      return std::nullopt;
    }

    p3d.image_ids.resize(track_length);
    p3d.point2D_idxs.resize(track_length);

    for (uint64_t j = 0; j < track_length; ++j) {
      if (!ReadBinary(file, p3d.image_ids[j]) || !ReadBinary(file, p3d.point2D_idxs[j])) {
        std::cerr << "Error: Failed to read track element." << std::endl;
        return std::nullopt;
      }
    }
    points3D[p3d.id] = std::move(p3d);
  }
  return points3D;
}
} // namespace colmap
