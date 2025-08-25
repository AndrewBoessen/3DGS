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
