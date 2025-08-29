#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Asserts that a pointer is a valid CUDA device pointer.
 * @param ptr The pointer to check.
 * @param file The source file where the assertion is made (usually __FILE__).
 * @param line The line number where the assertion is made (usually __LINE__).
 */
void assertIsDevicePointer(const void *ptr, const char *file, int line) {
  if (ptr == nullptr) {
    fprintf(stderr, "Assertion failed at %s:%d: Pointer is null.\n", file, line);
    exit(EXIT_FAILURE);
  }

  cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

  // An error here often means the pointer is invalid or not allocated by CUDA
  if (err != cudaSuccess) {
    fprintf(stderr, "Assertion failed at %s:%d: cudaPointerGetAttributes returned %s.\n", file, line,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // The 'type' field tells you where the memory resides
  if (attributes.type != cudaMemoryTypeDevice) {
    fprintf(stderr, "Assertion failed at %s:%d: Pointer is not a device pointer.\n", file, line);
    exit(EXIT_FAILURE);
  }
}

// A simple macro to make calling the assertion function easier
#define ASSERT_DEVICE_POINTER(p) assertIsDevicePointer(p, __FILE__, __LINE__)
