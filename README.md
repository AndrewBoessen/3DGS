# 3D Gaussian Splatting

![3D Gaussian Splatting](./assets/overview.jpg)

[![arXiv](https://img.shields.io/badge/arXiv-2308.04079-b31b1b.svg)](https://arxiv.org/abs/2308.04079)
[![Build Status](https://img.shields.io/github/actions/workflow/status/AndrewBoessen/3DGS/ci.yml?branch=main)](https://github.com/AndrewBoessen/3DGS/actions)
[![GitHub release](https://img.shields.io/github/v/release/AndrewBoessen/3DGS)](https://github.com/AndrewBoessen/3DGS/releases)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)
[![CUDA Version](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-downloads)

A minimalist CUDA and C++ implementation of **3D Gaussian Splatting** for real-time radiance field rendering.
This repository provides a lightweight foundation for experimenting with the core ideas from the original paper.

## Installing

### 1. Core Build Environment

You will need a standard C++ development toolchain. Please install the following using your system's package manager (e.g., `apt` for Debian/Ubuntu, `brew` for macOS).

- **C++ Compiler** with C++23 support (e.g., GCC 12+, Clang 15+)

- **CMake** (version 3.20 or later)

- **Eigen** (version 3.3 or later)

- **yaml-cpp**

- **sphericart**

For example, on Ubuntu 22.04, you can install these with:

```bash
git clone https://github.com/lab-cosmo/sphericart
cd sphericart/
mkdir build
cd build/
cmake ..
make install
```

```bash
sudo apt install cmake build-essential g++-12 libeigen3-dev libyaml-cpp-dev
```

### 2. NVIDIA CUDA Toolkit

A compatible NVIDIA driver and the CUDA Toolkit are required for the GPU-accelerated components.

- **NVIDIA CUDA Toolkit** (version 13.0 or later)

Please download and install the appropriate version for your system from the [**NVIDIA CUDA Toolkit website**](https://developer.nvidia.com/cuda-downloads). After installation, ensure that the `PATH`, `CPLUS_INCLUDE_PATH`, and `LD_LIBRARY_PATH` environment variables are configured correctly in your shell profile (e.g., `.bashrc`).

## Building

### 1. **Clone the repository:**

```bash
git clone https://github.com/AndrewBoessen/3DGS.git
cd 3DGS
```

### 2. **Configure with CMake:** Create a build directory and run CMake

```bash
cmake -S . -B build
```

_Note: If you have multiple compilers, you can specify one, e.g., `-DCMAKE_CXX_COMPILER=g++-12`._

### 3. **Compile the code:** Build the project using the number of available processor cores for faster compilation

```bash
cmake --build build --parallel $(nproc)
```

The main executable, `gaussian_splatting`, will be located in the `build/` directory.

## Using

![bicycle rendering](./assets/bicycle.gif)

### Mip-NeRF 360

### COLMAP

## Testing

The project includes a suite of unit tests built with Google Test. The tests are compiled automatically during the main build process.

To run all tests, execute `ctest` from within the build directory. The `--output-on-failure` flag is recommended to only show logs for failing tests.

```bash
cd build
ctest --output-on-failure
```

This will discover and run all test executables, including those for the CUDA kernels.
