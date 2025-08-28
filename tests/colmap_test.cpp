#include "dataloader/colmap.hpp"
#include "gtest/gtest.h"

const std::string TEST_DATA_PATH = TEST_DATA_DIR;

// Test suite for the ReadCamerasBinary function
TEST(ColmapIoTest, ReadCamerasBinary) {
  const auto cameras_optional = ReadCamerasBinary(TEST_DATA_PATH + "/cameras.bin");

  // 1. Ensure the file was opened and read successfully
  ASSERT_TRUE(cameras_optional.has_value());
  const auto &cameras = cameras_optional.value();

  // 2. Check the number of cameras
  ASSERT_EQ(cameras.size(), 1);
  ASSERT_TRUE(cameras.contains(1)); // Check if camera with ID 1 exists

  // 3. Verify the contents of the camera
  const auto &cam = cameras.at(1);
  EXPECT_EQ(cam.id, 1);
  EXPECT_EQ(cam.model, "SIMPLE_PINHOLE");
  EXPECT_EQ(cam.width, 100);
  EXPECT_EQ(cam.height, 80);
  ASSERT_EQ(cam.params.size(), 3);
  EXPECT_DOUBLE_EQ(cam.params[0], 150.5); // f
  EXPECT_DOUBLE_EQ(cam.params[1], 50.2);  // cx
  EXPECT_DOUBLE_EQ(cam.params[2], 40.8);  // cy
}

// Test suite for the ReadImagesBinary function
TEST(ColmapIoTest, ReadImagesBinary) {
  const auto images_optional = ReadImagesBinary(TEST_DATA_PATH + "/images.bin");

  ASSERT_TRUE(images_optional.has_value());
  const auto &images = images_optional.value();

  ASSERT_EQ(images.size(), 1);
  ASSERT_TRUE(images.contains(1));

  const auto &img = images.at(1);
  EXPECT_EQ(img.id, 1);
  EXPECT_EQ(img.name, "test.jpg");
  EXPECT_EQ(img.camera_id, 1);

  EXPECT_NEAR(img.qvec(0), 0.8, 1e-9); // w
  EXPECT_NEAR(img.tvec(0), 5.1, 1e-9); // x

  ASSERT_EQ(img.xys.size(), 2);
  EXPECT_DOUBLE_EQ(img.xys[0].x(), 10.1);
  EXPECT_DOUBLE_EQ(img.xys[0].y(), 11.2);
  EXPECT_EQ(img.point3D_ids[0], 1);
  EXPECT_EQ(img.point3D_ids[1], -1); // No 3D point
}

// Test suite for the ReadPoints3DBinary function
TEST(ColmapIoTest, ReadPoints3DBinary) {
  const auto points_optional = ReadPoints3DBinary(TEST_DATA_PATH + "/points3D.bin");

  ASSERT_TRUE(points_optional.has_value());
  const auto &points = points_optional.value();

  ASSERT_EQ(points.size(), 1);
  ASSERT_TRUE(points.contains(1));

  const auto &pt = points.at(1);
  EXPECT_EQ(pt.id, 1);
  EXPECT_DOUBLE_EQ(pt.xyz.x(), 1.1);
  EXPECT_DOUBLE_EQ(pt.xyz.y(), 2.2);
  EXPECT_DOUBLE_EQ(pt.xyz.z(), 3.3);
  EXPECT_EQ(pt.rgb[0], 10);
  EXPECT_EQ(pt.rgb[1], 20);
  EXPECT_EQ(pt.rgb[2], 30);
  EXPECT_NEAR(pt.error, 0.01, 1e-9);

  ASSERT_EQ(pt.image_ids.size(), 1);
  EXPECT_EQ(pt.image_ids[0], 1);
  EXPECT_EQ(pt.point2D_idxs[0], 0);
}
