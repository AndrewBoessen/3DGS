#include "dataloader/colmap.hpp"
#include "gsplat/utils.hpp"
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  // 1. Check for the correct number of command-line arguments.
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << "<path_to_config_file.yaml> <path_to_cameras.bin> <path_to_images.bin> <path_to_points3D.bin>"
              << std::endl;
    return 1; // Return an error code
  }

  // 2. Get file paths from the arguments.
  const std::string config_path = argv[1];
  const std::string cameras_path = argv[2];
  const std::string images_path = argv[3];
  const std::string points3D_path = argv[4];

  // Read config file and load parameters
  ConfigParameters config = parseConfig(config_path);

  std::cout << "Attempting to read COLMAP binary files..." << std::endl;
  std::cout << "Cameras file: " << cameras_path << std::endl;
  std::cout << "Images file: " << images_path << std::endl;
  std::cout << "Points3D file: " << points3D_path << std::endl;
  std::cout << std::endl;

  // 3. Read the cameras.bin file.
  const auto cameras_optional = ReadCamerasBinary(cameras_path);
  if (cameras_optional) {
    const auto &cameras = cameras_optional.value();
    std::cout << "Successfully read " << cameras.size() << " cameras." << std::endl;
  } else {
    std::cerr << "Error: Could not read cameras file at " << cameras_path << std::endl;
    return 1;
  }

  // 4. Read the images.bin file.
  const auto images_optional = ReadImagesBinary(images_path);
  if (images_optional) {
    const auto &images = images_optional.value();
    std::cout << "Successfully read " << images.size() << " images." << std::endl;
  } else {
    std::cerr << "Error: Could not read images file at " << images_path << std::endl;
    return 1;
  }

  // 5. Read the points3D.bin file.
  const auto points_optional = ReadPoints3DBinary(points3D_path);
  if (points_optional) {
    const auto &points = points_optional.value();
    std::cout << "Successfully read " << points.size() << " 3D points." << std::endl;
  } else {
    std::cerr << "Error: Could not read points3D file at " << points3D_path << std::endl;
    return 1;
  }

  return 0; // Success
}
