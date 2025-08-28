# CMakeLists.txt

cmake_minimum_required(VERSION 3.20)
project(ColmapReader)

# Use C++23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Eigen3 library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# Find YAML library
find_package(yaml-cpp REQUIRED)

# ===================================================================
# DATALODER LIBRARY DEFINITION
# ===================================================================
add_library(dataloader src/colmap.cpp)

target_include_directories(dataloader
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${EIGEN3_INCLUDE_DIR}
)

# ===================================================================
# GSPLAT LIBRARY DEFINITION
# ===================================================================
add_library(gsplat src/utils.cpp)

target_include_directories(gsplat
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(gsplat PUBLIC yaml-cpp::yaml-cpp)

# ===================================================================
# MAIN EXECUTABLE
# ===================================================================
add_executable(gaussian_splatting src/main.cpp)

# Link the main executable against our library
target_link_libraries(gaussian_splatting PRIVATE dataloader gsplat)

# ===================================================================
# TESTING SETUP
# ===================================================================

# Enable testing for the project
enable_testing()

# Download and configure GoogleTest using FetchContent
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# Make the googletest targets available
FetchContent_MakeAvailable(googletest)

# Add the test executable
add_executable(colmap_tests tests/colmap_test.cpp)

# Pass the full path to the test data directory
target_compile_definitions(colmap_tests PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test_data")

# Link the test executable against our library and GoogleTest
target_link_libraries(colmap_tests PRIVATE dataloader gtest_main)

# Add the test to CTest
include(GoogleTest)
gtest_discover_tests(colmap_tests)
