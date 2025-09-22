// gaussian.cpp

#include "gsplat/gaussian.hpp"
#include <cassert>
#include <cmath>
#include <nanoflann.hpp>
#include <omp.h>
#include <vector>

// A simple adaptor for nanoflann to interface with our Point3D vector.
// This avoids copying the data into a different structure.
struct PointCloudAdaptor {
  const std::vector<Point3D> &points;

  PointCloudAdaptor(const std::vector<Point3D> &vec) : points(vec) {}

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return points.size(); }

  // Returns the squared distance between a query point and the data point at a given index.
  inline double kdtree_distance(const double *p1, const size_t idx_p2, size_t /*size*/) const {
    const double d0 = p1[0] - points[idx_p2].xyz.x();
    const double d1 = p1[1] - points[idx_p2].xyz.y();
    const double d2 = p1[2] - points[idx_p2].xyz.z();
    return d0 * d0 + d1 * d1 + d2 * d2;
  }

  // Returns the value of a single dimension of a point at a given index.
  inline double kdtree_get_pt(const size_t idx, int dim) const { return points[idx].xyz[dim]; }

  // Optional bounding-box computation (not needed here, so we return false).
  template <class BBOX> bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }
};

/**
 * @brief Initialize Gaussians on points in Point3D.
 */
Gaussians Gaussians::Initialize(const std::unordered_map<uint64_t, Point3D> &points) {
  size_t num_points = points.size();
  if (num_points == 0) {
    return Gaussians();
  }

  // Pre-allocate and resize memory for direct indexed access in the parallel loop
  std::vector<Eigen::Vector3f> xyz_vec(num_points);
  std::vector<Eigen::Vector3f> rgb_vec(num_points);
  std::vector<float> opacity_vec(num_points);
  std::vector<Eigen::Vector3f> scale_vec(num_points);
  std::vector<Eigen::Quaternionf> quaternion_vec(num_points);

  // Copy points to a vector for indexed access
  std::vector<Point3D> point_vec;
  point_vec.reserve(num_points);
  for (const auto &[id, point] : points) {
    point_vec.push_back(point);
  }

  // Build the k-d tree index for our point cloud
  PointCloudAdaptor cloud_adaptor(point_vec);
  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor>,
                                                     PointCloudAdaptor, 3 /* 3D points */>;
  KDTree index(3, cloud_adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
  index.buildIndex();

// For each point, find its nearest neighbors using the k-d tree.
#pragma omp parallel for
  for (size_t i = 0; i < num_points; ++i) {
    const auto &current_point = point_vec[i];

    // We search for 4 neighbors because the point itself will be the closest one (distance = 0).
    const size_t num_neighbors_to_find = 4;
    std::vector<size_t> ret_indexes(num_neighbors_to_find);
    std::vector<double> out_dists_sq(num_neighbors_to_find);

    nanoflann::KNNResultSet<double> resultSet(num_neighbors_to_find);
    resultSet.init(&ret_indexes[0], &out_dists_sq[0]);
    index.findNeighbors(resultSet, &current_point.xyz[0], nanoflann::SearchParameters(10));

    // Calculate the average distance, skipping the first result (the point itself)
    double total_dist = 0.0;
    size_t count = 0;
    // Start from k=1 to ignore the self-match
    for (size_t k = 1; k < num_neighbors_to_find && k < resultSet.size(); ++k) {
      total_dist += std::sqrt(out_dists_sq[k]);
      count++;
    }
    const float avg_dist = (count > 0) ? static_cast<float>(total_dist / count) : 0.01f;

    // Initialize the Gaussian properties directly into the pre-sized vectors
    xyz_vec[i] = current_point.xyz.cast<float>();
    rgb_vec[i] = Eigen::Vector3f(current_point.rgb[0], current_point.rgb[1], current_point.rgb[2]) / 255.0f;
    opacity_vec[i] = 0.1f;
    scale_vec[i] = Eigen::Vector3f(logf(avg_dist), logf(avg_dist), logf(avg_dist));
    quaternion_vec[i] = Eigen::Quaternionf::Identity();
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
