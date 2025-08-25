# CMakeLists.txt

cmake_minimum_required(VERSION 3.20)
project(ColmapReader)

# Use C++23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Eigen3 library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(colmap src/colmap.cpp)

target_include_directories(colmap
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${EIGEN3_INCLUDE_DIR}
)

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
target_link_libraries(colmap_tests PRIVATE colmap gtest_main)

# Add the test to CTest
include(GoogleTest)
gtest_discover_tests(colmap_tests)
