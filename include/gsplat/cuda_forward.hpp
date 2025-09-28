// cuda_functions.hpp

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// Macro for checking CUDA API calls for errors.
#define CHECK_CUDA(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));               \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

/**
 * @brief Compute conic of projected 2D covariance matrix
 * @param[in]  xyz    A device pointer to 3D points
 * @param[in]  K      Camera intrinsic projection matrix
 * @param[in]  sigma  3D Gaussian covariance matrix
 * @param[in]  T      Camera extrinsic projection matrix
 * @param[in]  N      The total number of points
 * @param[out] J      A device pointer to ouput Jacobian
 * @param[out] conic  A device pointer to output conic values
 * @param[in]  stream The CUDA stream to execute kernel on
 */
void compute_conic(float *const xyz, const float *K, float *const sigma, const float *T, const int N, float *J,
                   float *conic, cudaStream_t stream = 0);

/**
 * @brief Compute covariance matrix of Gaussian from quaternion and scale vector
 * @param[in]  quaternion  A device pointer to Gaussian quaternion
 * @param[in]  scale       A device pointer to scale vectors
 * @param[in]  N           The total number of Gaussians
 * @param[out] sigma       A device pointer to ouput covariance matrix
 * @param[in]  stream The CUDA stream to execute kernel on
 */
void compute_sigma(float *const quaternion, float *const scale, const int N, float *sigma, cudaStream_t stream = 0);

/**
 * @brief Compute camera view of points from rotation matrix and translation vector
 * @param[in]  xyz_w  A device pointer to world view of points
 * @param[in]  T      A device pointer to camera extrinsic matrix
 * @param[in]  N      The total number of points
 * @param[out] xyz_c  A device porinter to output camera view
 * @param[in]  stream The CUDA stream to execute kernel on
 */
void camera_extrinsic_projection(float *const xyz_w, const float *T, const int N, float *xyz_c,
                                 cudaStream_t stream = 0);

/**
 * @brief Launches the CUDA kernel for projecting 3D points to 2D image coordinates.
 * @param[in]  xyz  A device pointer to the input array of 3D points.
 * @param[in]  K    A device pointer to the camera intrinsic matrix.
 * @param[in]  N    The total number of points.
 * @param[out] uv   A device pointer to the output array for 2D coordinates.
 * @param[in]  stream The CUDA stream to execute kernel on
 */
void camera_intrinsic_projection(float *const xyz, const float *K, const int N, float *uv, cudaStream_t stream = 0);

/**
 * @brief Lauches CUDA kernel to perform frustum culling on guassians.
 * @param[in]  uv           A device pointer to gaussian coordinates in image frame
 * @param[in]  xyz          A device pointer to 3D corrdinates of gaussians in camera perspective
 * @param[in]  N            Number of Gaussians
 * @param[in]  near_thresh  Distance to cull guassians closer than threshold
 * @param[in]  far_thresh   Distance to cull gaussians farther than threshold
 * @param[in]  padding      Padding distance beyond image frame
 * @param[in]  width        Image width
 * @param[in]  height       Image height
 * @param[out] mask         A device pointer for mask on guassians
 * @param[in]  stream The CUDA stream to execute kernel on
 */
void cull_gaussians(float *const uv, float *const xyz, const int N, const float near_thresh, const float far_thresh,
                    const int padding, const int width, const int height, bool *mask, cudaStream_t stream = 0);

/**
 * @brief Filter the Guassian parameters by the provided mask and return filtered arrays
 * @param[in]  N
 * @param[in]  d_mask
 * @param[in]  d_xyz
 * @param[in]  d_rgb
 * @param[in]  d_opacity
 * @param[in]  d_scale
 * @param[in]  d_quaternion
 * @param[in]  d_uv
 * @param[in]  d_xyz_c
 * @param[out] d_xyz_culled
 * @param[out] d_rgb_culled
 * @param[out] d_opacity_culled
 * @param[out] d_scale_culled
 * @param[out] d_quaternion_culled
 * @param[out] d_uv_culled
 * @param[out] d_xyz_c_culled
 * @param[out] h_num_culled
 * @param[in]  stream The CUDA stream to execute kernel on
 */
void filter_gaussians_by_mask(int N, const bool *d_mask, const float *d_xyz, const float *d_rgb, const float *d_opacity,
                              const float *d_scale, const float *d_quaternion, const float *d_uv, const float *d_xyz_c,
                              float *d_xyz_culled, float *d_rgb_culled, float *d_opacity_culled, float *d_scale_culled,
                              float *d_quaternion_culled, float *d_uv_culled, float *d_xyz_c_culled, int *h_num_culled,
                              cudaStream_t stream = 0);

/**
 * @brief Lanuches CUDA kernels to get gaussian tile intersections sorted by depth
 * @param[in]  uv                               A device pointer to gaussian coordinates in image frame
 * @param[in]  xyz                              A device pointer to 3D corrdinates of gaussians in camera perspective
 * @param[in]  conic                            A device pointer to 2D gaussian conic
 * @param[in]  n_tiles_x                        Number of tiles in image x axis
 * @param[in]  n_tiles_y                        Number of tiles in image y axis
 * @param[in]  mh_dist                          Mahalanobis distance to define bounding box
 * @param[in]  N                                The total number of points
 * @param[out] sorted_gaussian_bytes            Pointer to store bytes to allocate for sorted_gaussians
 * @param[out] sorted_gaussians                 A device array to ouput gaussians sorted by z depth
 * @param[out] splat_start_end_idx_by_tile_idx  A device array to index into sorted_gaussian by tile id
 * @param[in]  stream                           The CUDA stream to execute kernel on
 */
void get_sorted_gaussian_list(const float *uv, const float *xyz, const float *conic, const int n_tiles_x,
                              const int n_tiles_y, const float mh_dist, const int N, size_t &sorted_gaussian_bytes,
                              int *sorted_gaussians, int *splat_start_end_idx_by_tile_idx, cudaStream_t stream = 0);

/**
 * @brief Launches CUDA kernels to precompute spherical harmonic values and calculate rgb values
 * @param[in]  xyz              A device pointer to 3D corrdinates of gaussians in camera perspective
 * @param[in]  sh_coefficients  A device pointer to SH params for each Gaussian
 * @param[in]  l_max            The max degree of SH
 * @param[in]  N                The total number of points
 * @param[out] rgb              A device pointer to output rgb values
 * @param[in]  stream           The CUDA stream to execute kernel on
 */
void precompute_spherical_harmonics(const float *xyz, const float *sh_coefficients, const int N, const int max_l,
                                    float *rgb, cudaStream_t stream = 0);

/**
 * @brief Launch CUDA kernels to render image pixel values from Gaussians
 * @param[in]  uv                   A device pointer to centers of splats
 * @param[in]  opacity              A device pointer to splat opacity values
 * @param[in]  conic                A device pointer to store splat conic parameters
 * @param[in]  rgb                  A device pointer to precomputed rgb values
 * @param[in]  background_opacity   The opacity value to use for white background
 * @param[in]  sorted_splats        A device pointer to Gaussian ids for each tile
 * @param[in]  splat_range_by_tile  A device pointer to start and end ids into sorted_splats
 * @param[in]  image_width          The width of image in pixels
 * @param[in]  image_height         The height of image in pixels
 * @param[in]  stream The CUDA stream to execute kernel on
 * @param[out] weight_per_pixel     A device pointer to output final alpha weights per pixel
 * @param[out] image                A device pointer to output image rgb values
 * @param[in]  stream               The CUDA stream to execute kernel on
 */
void render_image(const float *uv, const float *opacity, const float *conic, const float *rgb,
                  const float background_opacity, const int *sorted_splats, const int *splat_range_by_tile,
                  const int image_width, const int image_height, float *weight_per_pixel, float *image,
                  cudaStream_t stream = 0);

/**
 * @brief Launch CUDA kernel to compute L1 and SSIM loss with gradient output
 * @param[in]  predicted_data  A device pointer to predicited image
 * @param[in]  gt_data         A device pointer to ground truth image
 * @param[in]  rows            Height of the image
 * @param[in]  cols            Width of the image
 * @param[in]  channels        Number of channels (3 for RGB)
 * @param[in]  ssim_weight     Alpha value in loss function
 * @param[out] image_grad      Gradient per pixel channel
 * @param[in]  stream          The CUDA stream to execute kernel on
 * @return Loss from combined L1 and SSIM
 */
float fused_loss(const float *predicted_data, const float *gt_data, int rows, int cols, int channels,
                 const float ssim_weight, float *image_grad, cudaStream_t stream = 0);
