// gaussian.cpp

#include "gsplat/gaussian.hpp"

#include <cassert>
#include <cmath>
#include <queue>
#include <vector>

/**
 * @brief Initialize Gaussians on points in Point3D.
 */
Gaussians Gaussians::Initialize(const std::unordered_map<uint64_t, Point3D> &points) {
  size_t num_points = points.size();
  if (num_points == 0) {
    return Gaussians();
  }

  // Pre-allocate memory for efficiency
  std::vector<Eigen::Vector3f> xyz_vec;
  xyz_vec.reserve(num_points);
  std::vector<Eigen::Vector3f> rgb_vec;
  rgb_vec.reserve(num_points);
  std::vector<float> opacity_vec;
  opacity_vec.reserve(num_points);
  std::vector<Eigen::Vector3f> scale_vec;
  scale_vec.reserve(num_points);
  std::vector<Eigen::Quaternionf> quaternion_vec;
  quaternion_vec.reserve(num_points);

  // Copy points to a vector for indexed access, which is required for nearest neighbor search
  std::vector<Point3D> point_vec;
  point_vec.reserve(num_points);
  for (const auto &[id, point] : points) {
    point_vec.push_back(point);
  }

  // For each point, find the 3 nearest neighbors and compute the average distance
  for (size_t i = 0; i < num_points; ++i) {
    const auto &current_point = point_vec[i];

    // A max-heap to efficiently keep track of the 3 smallest distances found so far.
    // We store squared distances to avoid costly sqrt operations until the very end.
    std::priority_queue<double> nearest_distances_sq;

    for (size_t j = 0; j < num_points; ++j) {
      if (i == j)
        continue;

      const auto &other_point = point_vec[j];
      double dist_sq = (current_point.xyz - other_point.xyz).squaredNorm();

      if (nearest_distances_sq.size() < 3) {
        nearest_distances_sq.push(dist_sq);
      } else if (dist_sq < nearest_distances_sq.top()) {
        nearest_distances_sq.pop();
        nearest_distances_sq.push(dist_sq);
      }
    }

    // Calculate the average distance to the 3 nearest neighbors
    double total_dist = 0.0;
    const size_t count = nearest_distances_sq.size();
    while (!nearest_distances_sq.empty()) {
      total_dist += std::sqrt(nearest_distances_sq.top());
      nearest_distances_sq.pop();
    }
    // Use a default small scale if no neighbors are found (e.g., for a single point)
    const float avg_dist = (count > 0) ? static_cast<float>(total_dist / count) : 0.01f;

    // Initialize the Gaussian properties
    xyz_vec.push_back(current_point.xyz.cast<float>());
    rgb_vec.push_back(Eigen::Vector3f(current_point.rgb[0], current_point.rgb[1], current_point.rgb[2]) / 255.0f);
    opacity_vec.push_back(0.1f);
    scale_vec.push_back(Eigen::Vector3f(avg_dist, avg_dist, avg_dist)); // New scale initialization
    quaternion_vec.push_back(Eigen::Quaternionf::Identity());
  }

  // Construct the Gaussians object by moving the created vectors
  // Note: SH is left as std::nullopt by default.
  return Gaussians(std::move(xyz_vec), std::move(rgb_vec), std::move(opacity_vec), std::move(scale_vec),
                   std::move(quaternion_vec));
}

void Gaussians::append(const Gaussians &other) {
  if (other.size() == 0) {
    return;
  }

  // Append core attributes
  xyz.insert(xyz.end(), other.xyz.begin(), other.xyz.end());
  rgb.insert(rgb.end(), other.rgb.begin(), other.rgb.end());
  opacity.insert(opacity.end(), other.opacity.begin(), other.opacity.end());
  scale.insert(scale.end(), other.scale.begin(), other.scale.end());
  quaternion.insert(quaternion.end(), other.quaternion.begin(), other.quaternion.end());

  // Handle optional Spherical Harmonics with the new structure
  if (sh.has_value() && other.sh.has_value()) {
    sh->insert(sh->end(), other.sh->begin(), other.sh->end());
  } else {
    sh = std::nullopt;
  }
}

void Gaussians::filter(const std::vector<bool> &mask) {
  const size_t original_size = size();
  assert(mask.size() == original_size && "Mask size must match the number of Gaussians.");

  if (original_size == 0) {
    return;
  }

  size_t write_idx = 0;
  for (size_t read_idx = 0; read_idx < original_size; ++read_idx) {
    if (mask[read_idx]) {
      if (write_idx != read_idx) {
        xyz[write_idx] = std::move(xyz[read_idx]);
        rgb[write_idx] = std::move(rgb[read_idx]);
        opacity[write_idx] = std::move(opacity[read_idx]);
        scale[write_idx] = std::move(scale[read_idx]);
        quaternion[write_idx] = std::move(quaternion[read_idx]);

        // Filter SH exactly like the other properties
        if (sh.has_value()) {
          (*sh)[write_idx] = std::move((*sh)[read_idx]);
        }
      }
      write_idx++;
    }
  }

  // Truncate all vectors to their new, smaller size
  xyz.resize(write_idx);
  rgb.resize(write_idx);
  opacity.resize(write_idx);
  scale.resize(write_idx);
  quaternion.resize(write_idx);

  // Truncate the SH vector as well
  if (sh.has_value()) {
    sh->resize(write_idx);
  }
}
