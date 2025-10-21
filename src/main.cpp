#include "dataloader/colmap.hpp"
#include "gsplat/gaussian.hpp"
#include "gsplat/trainer.hpp"
#include "gsplat/utils.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

int main(int argc, char *argv[]) {
  // Check for the correct number of command-line arguments.
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " <scaling factor> <path_to_config_file.yaml> <path_to_cameras.bin> <path_to_images.bin> "
                 "<path_to_points3D.bin> "
                 "<path_to_image_root_directory>"
              << std::endl;
    return 1; // Return an error code
  }

  // Get file paths from the arguments.
  const int downsample_factor = *argv[1] - '0';
  const std::string config_path = argv[2];
  const std::string cameras_path = argv[3];
  const std::string images_path = argv[4];
  const std::string points3D_path = argv[5];
  const std::string image_dir = argv[6];

  // --- 1. Declarations ---
  // All data is declared here to be accessible at the end of the main function.
  ConfigParameters config;

  using CamerasType = decltype(ReadCamerasBinary({}, 1))::value_type;
  using ImagesType = decltype(ReadImagesBinary({}, "", 1))::value_type;
  using Points3DType = decltype(ReadPoints3DBinary({}))::value_type;

  CamerasType cameras;
  ImagesType images;
  Points3DType points;
  Gaussians gaussians;

  // --- 2. Data Loading ---
  // Read config file and load parameters
  std::cout << "Attempting to read " << config_path << std::endl;
  try {
    config = parseConfig(config_path);
  } catch (const std::runtime_error &e) {
    std::cout << "Failed to load config file: " << e.what() << std::endl;
    return 1; // Return on error
  }
  std::cout << "Successfully loaded config file" << std::endl;

  std::cout << "\nAttempting to read COLMAP binary files..." << std::endl;
  std::cout << "Cameras file: " << cameras_path << std::endl;
  std::cout << "Images file: " << images_path << std::endl;
  std::cout << "Points3D file: " << points3D_path << std::endl;

  // Read the cameras.bin file.
  if (auto cameras_optional = ReadCamerasBinary(cameras_path, downsample_factor)) {
    cameras = std::move(*cameras_optional); // Move data from optional to main variable
    std::cout << "Successfully read " << cameras.size() << " cameras." << std::endl;
  } else {
    std::cerr << "Error: Could not read cameras file at " << cameras_path << std::endl;
    return 1;
  }

  // Read the images.bin file.
  if (auto images_optional = ReadImagesBinary(images_path, image_dir, downsample_factor)) {
    images = std::move(*images_optional);
    std::cout << "Successfully read " << images.size() << " images." << std::endl;
  } else {
    std::cerr << "Error: Could not read images file at " << images_path << std::endl;
    return 1;
  }

  // Read the points3D.bin file.
  if (auto points_optional = ReadPoints3DBinary(points3D_path)) {
    points = std::move(*points_optional);
    std::cout << "Successfully read " << points.size() << " 3D points." << std::endl;
  } else {
    std::cerr << "Error: Could not read points3D file at " << points3D_path << std::endl;
    return 1;
  }

  // --- 3. Initialization ---
  // Initialize Gaussians with the loaded 3D points
  gaussians = Gaussians::Initialize(points);
  std::cout << "Successfully initialized " << gaussians.xyz.size() << " Gaussians." << std::endl;

  return 0; // Success
}
