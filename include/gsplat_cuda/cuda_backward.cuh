// cuda_backward.hpp

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

inline constexpr int TILE_SIZE_BWD = 16;

/**
 * @brief Compute gradients for the camera intrinsic projection.
 * @param[in]  xyz_c          A device pointer to 3D points in camera coordinates.
 * @param[in]  proj           A device pointer to the camera projection matrix (4x4).
 * @param[in]  uv_grad_out    A device pointer to the upstream gradients from the 2D projection.
 * @param[in]  N              The total number of points.
 * @param[in]  width          Image width.
 * @param[in]  height         Image height.
 * @param[out] xyz_c_grad_in  A device pointer to store the computed gradients for xyz_c.
 * @param[in]  stream         The CUDA stream to execute the kernel on.
 */
void project_to_screen_backward(const float *const xyz_c, const float *const proj, const float *const uv_grad_out,
                                const int N, const int width, const int height, float *xyz_c_grad_in,
                                cudaStream_t stream = 0);

/**
 * @brief Compute gradients for the camera extrinsic transformation.
 * @param[in]  xyz_w           A device pointer to 3D points in world coordinates.
 * @param[in]  view            A device pointer to the camera view matrix (4x4).
 * @param[in]  xyz_c_grad_out  A device pointer to the upstream gradients from camera-space coordinates.
 * @param[in]  N               The total number of points.
 * @param[out] xyz_w_grad_in   A device pointer to store the computed gradients for xyz_w.
 * @param[in]  stream          The CUDA stream to execute the kernel on.
 */
void compute_camera_space_points_backward(const float *const xyz_w, const float *const view,
                                          const float *const xyz_c_grad_out, const int N, float *xyz_w_grad_in,
                                          cudaStream_t stream = 0);

/**
 * @brief Compute gradients for the projection Jacobian.
 * @param[in]  xyz_c            A device pointer to 3D points in camera coordinates.
 * @param[in]  proj             A device pointer to the camera projection matrix (4x4).
 * @param[in]  J_grad_out       A device pointer to the upstream gradients for the Jacobian J.
 * @param[in]  N                The total number of points.
 * @param[out] xyz_c_grad_in    A device pointer to store the computed gradients for xyz_c.
 * @param[in]  stream           The CUDA stream to execute the kernel on.
 */
void compute_projection_jacobian_backward(const float *const xyz_c, const float *const proj,
                                          const float *const J_grad_out, const int N, float *xyz_c_grad_in,
                                          cudaStream_t stream = 0);

/**
 * @brief Compute gradients for the 2D conic projection.
 * @param[in]  J                A device pointer to the projection Jacobians.
 * @param[in]  sigma            A device pointer to the 3D covariance matrices.
 * @param[in]  view             A device pointer to the camera view matrix (4x4).
 * @param[in]  conic_grad_out   A device pointer to the upstream gradients for the conic.
 * @param[in]  N                The total number of points.
 * @param[out] J_grad_in        A device pointer to store the computed gradients for J.
 * @param[out] sigma_grad_in    A device pointer to store the computed gradients for sigma.
 * @param[in]  stream           The CUDA stream to execute the kernel on.
 */
void compute_conic_backward(const float *const J, const float *const sigma, const float *const view,
                            const float *const conic, const float *const conic_grad_out, const int N, float *J_grad_in,
                            float *sigma_grad_in, cudaStream_t stream = 0);

/**
 * @brief Compute gradients for the 3D covariance matrix (sigma).
 * @param[in]  quaternion          A device pointer to the quaternions (w, x, y, z).
 * @param[in]  scale               A device pointer to the scale factors (sx, sy, sz).
 * @param[in]  sigma_grad_out      A device pointer to the upstream gradients for sigma.
 * @param[in]  N                   The total number of points.
 * @param[out] quaternion_grad_in  A device pointer to store the computed gradients for the quaternions.
 * @param[out] scale_grad_in       A device pointer to store the computed gradients for the scales.
 * @param[in]  stream              The CUDA stream to execute the kernel on.
 */
void compute_sigma_backward(const float *const quaternion, const float *const scale, const float *const sigma_grad_out,
                            const int N, float *quaternion_grad_in, float *scale_grad_in, cudaStream_t stream = 0);

/**
 * @brief Compute gradients for the spherical harmonic coefficients
 * @param[in]  xyz_c              Camera xyz coordinates
 * @param[in]  rgb_grad_out       RGB gradients
 * @param[in]  l_max              The max degree of SH
 * @param[in]  N                  The total number of points
 * @param[out] sh_grad_in         Spherical harmonic gradients
 * @param[out] sh_grad_band_0_in  Spherical harmonic gradients
 * @param[in]  stream             The CUDA stream to execute the kernel on.
 */
void precompute_spherical_harmonics_backward(const float *const xyz_c, const float *const rgb_grad_out, const int l_max,
                                             const int N, float *sh_grad_in, float *sh_grad_band_0_in,
                                             cudaStream_t stream = 0);

/**
 * @brief Launch the CUDA kernel to compute rendering gradients.
 * @param[in]  uvs                     Device pointer to 2D projected means.
 * @param[in]  opacity                 Device pointer to opacities.
 * @param[in]  conic                   Device pointer to 2D conic matrices.
 * @param[in]  rgb                     Device pointer to SH coefficients.
 * @param[in]  background_opacity      Background opacity used in formward pass.
 * @param[in]  sorted_splats           Device pointer to sorted splat indices.
 * @param[in]  splat_range_by_tile     Device pointer to the start/end splat index for each tile.
 * @param[in]  num_splats_per_pixel    Device pointer to the number of splats affecting each pixel.
 * @param[in]  final_weight_per_pixel  Device pointer to the final alpha weight from the forward pass.
 * @param[in]  grad_image              Device pointer to the upstream gradients from the rendered image.
 * @param[in]  image_width             Width of the image in pixels.
 * @param[in]  image_height            Height of the image in pixels.
 * @param[out] grad_rgb                Device pointer for storing SH coefficient gradients.
 * @param[out] grad_opacity            Device pointer for storing opacity gradients.
 * @param[out] grad_uv                 Device pointer for storing 2D mean gradients.
 * @param[out] grad_conic              Device pointer for storing 2D conic gradients.
 * @param[in]  stream                  The CUDA stream for execution.
 */
void render_image_backward(const float *const uvs, const float *const opacity, const float *const conic,
                           const float *const rgb, const float background_opacity, const int *const sorted_splats,
                           const int *const splat_range_by_tile, const int *const num_splats_per_pixel,
                           const float *const final_weight_per_pixel, const float *const grad_image,
                           const int image_width, const int image_height, float *grad_rgb, float *grad_opacity,
                           float *grad_uv, float *grad_conic, cudaStream_t stream = 0);
