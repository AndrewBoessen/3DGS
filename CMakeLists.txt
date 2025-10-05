# CMakeLists.txt

cmake_minimum_required(VERSION 3.20)
project(GuassianSplattingProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

# Find the Eigen3 library
find_package(Eigen3 REQUIRED NO_MODULE)
# Find YAML library
find_package(yaml-cpp REQUIRED)
# Find CUDA Toolkit and its libraries (cuBLAS, etc.)
find_package(CUDAToolkit REQUIRED)
# Find sphericart library
find_package(sphericart REQUIRED)
# Find OpenMP for sphericart parallelization
find_package(OpenMP REQUIRED)
# Find nanoflann for nearest neighbor
find_package(nanoflann)
# Find OpenCV for image procesing
find_package(OpenCV REQUIRED)

# ===================================================================
# DATALODER LIBRARY DEFINITION
# ===================================================================
add_library(dataloader src/colmap.cpp)

target_include_directories(dataloader
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(dataloader PUBLIC Eigen3::Eigen)

# ===================================================================
# GSPLAT LIBRARY DEFINITION
# ===================================================================
add_library(gsplat src/utils.cpp src/data.cpp src/gaussian.cpp src/raster.cpp src/trainer.cpp)

target_include_directories(gsplat
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(gsplat PUBLIC dataloader gsplat_cuda yaml-cpp Eigen3::Eigen nanoflann::nanoflann ${OpenCV_LIBS})

# ===================================================================
# GSPLAT CUDA LIBRARY DEFINITION
# ===================================================================
add_library(gsplat_cuda cuda/gaussian.cu cuda/projection.cu cuda/culling.cu cuda/spherical_harmonics.cu cuda/render.cu cuda/loss.cu cuda/projection_backward.cu cuda/gaussian_backward.cu cuda/spherical_harmonics_backward.cu cuda/render_backward.cu cuda/optimizer.cu)

target_include_directories(gsplat_cuda
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(gsplat_cuda PUBLIC CUDA::cublas sphericart OpenMP::OpenMP_CXX)

# ===================================================================
# MAIN EXECUTABLE
# ===================================================================
add_executable(gaussian_splatting src/main.cpp)

# Link the main executable against our library
target_link_libraries(gaussian_splatting PRIVATE dataloader gsplat gsplat_cuda)

# ===================================================================
# TESTING SETUP
# ===================================================================
enable_testing()
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# Helper function to reduce boilerplate when adding a new GTest target.
# Usage: add_gtest_executable(<test_name> <source_file> [LIBS...])
function(add_gtest_executable test_name source_file)
  add_executable(${test_name} ${source_file})
  target_link_libraries(${test_name} PRIVATE gtest_main ${ARGN})
  gtest_discover_tests(${test_name})
endfunction()

# --- Dataloader Tests ---
add_gtest_executable(dataloader_tests tests/colmap_test.cpp dataloader)
target_compile_definitions(dataloader_tests PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test_data")

# --- Utils Tests ---
add_gtest_executable(utils_tests tests/utils_test.cpp gsplat)

# --- Gaussians Tests ---
add_gtest_executable(gaussian_tests tests/gaussian_test.cpp gsplat dataloader)

# --- CUDA Forward Tests ---
add_gtest_executable(cuda_forward_tests tests/cuda_forward_test.cpp gsplat_cuda)

# --- CUDA Backward Tests ---
add_gtest_executable(cuda_backward_tests tests/cuda_backward_test.cpp gsplat_cuda)

# --- CUDA Optimizer Tests ---
add_gtest_executable(cuda_optimizer_tests tests/optimizer_test.cpp gsplat_cuda)
