// loss.cu

#include "checks.cuh"
#include "gsplat_cuda/cuda_forward.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <thrust/device_vector.h>

namespace cg = cooperative_groups;

// ------------------------------------------
// Constant Memory for Gaussian Coefficients
// ------------------------------------------
__constant__ float cGauss[11] = {0.001028380123898387f,  0.0075987582094967365f, 0.036000773310661316f,
                                 0.10936068743467331f,   0.21300552785396576f,   0.26601171493530273f,
                                 0.21300552785396576f,   0.10936068743467331f,   0.036000773310661316f,
                                 0.0075987582094967365f, 0.001028380123898387f};

namespace SSIMConstants {
constexpr float K1 = 0.01f;
constexpr float K2 = 0.03f;
constexpr float L = 1.0f;
constexpr float C1 = (K1 * L) * (K1 * L);
constexpr float C2 = (K2 * L) * (K2 * L);
} // namespace SSIMConstants

// ------------------------------------------
// Block and Shared Memory Dimensions
// ------------------------------------------
#define BLOCK_X 16
#define BLOCK_Y 16
#define HALO 5
#define SHARED_X (BLOCK_X + 2 * HALO)
#define SHARED_Y (BLOCK_Y + 2 * HALO)
#define CONV_X BLOCK_X
#define CONV_Y SHARED_Y
#define CHANNELS 3

// ------------------------------------------
// Utility: Safe pixel fetch w/ Clamping for RGB
// ------------------------------------------
__device__ __forceinline__ float get_pix_value_clamped_rgb(const float *img, int y, int x, int c, int H, int W) {
  int cy = max(0, min(H - 1, y));
  int cx = max(0, min(W - 1, x));
  // Access interleaved data: (Pixel Index * 3) + Channel
  return img[(cy * W + cx) * CHANNELS + c];
}

// ------------------------------------------
// Forward Kernel: Fused SSIM + L1 Loss (RGB)
// ------------------------------------------
__global__ void fused_loss_forward_kernel(int H, int W, float C1, float C2, float ssim_weight,
                                          const float *__restrict__ pred, const float *__restrict__ gt,
                                          float *__restrict__ total_loss_ptr, float *__restrict__ dm_dmu1,
                                          float *__restrict__ dm_dsigma1_sq, float *__restrict__ dm_dsigma12) {
  auto block = cg::this_thread_block();

  // 1. Identify location
  const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
  const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
  const int pix_id = pix_y * W + pix_x;

  // Shared memory for the tile (pred, gt)
  // [Y][X][RGB][2 (pred/gt)]
  __shared__ float sTile[SHARED_Y][SHARED_X][CHANNELS][2];

  // After horizontal pass, store partial sums here
  // xconv[y][x][channel][stats]
  // Stats: (sumX, sumX^2, sumY, sumY^2, sumXY)
  __shared__ float xconv[CONV_Y][CONV_X][CHANNELS][5];

  // ------------------------------------------------------------
  // 2) Load (pred, gt) tile + halo into shared memory
  // ------------------------------------------------------------
  {
    const int tileSize = SHARED_Y * SHARED_X;
    const int threads = BLOCK_X * BLOCK_Y;
    const int steps = (tileSize + threads - 1) / threads;

    const int tileStartY = block.group_index().y * BLOCK_Y;
    const int tileStartX = block.group_index().x * BLOCK_X;

    for (int s = 0; s < steps; ++s) {
      int tid = s * threads + block.thread_rank();
      if (tid < tileSize) {
        int local_y = tid / SHARED_X;
        int local_x = tid % SHARED_X;
        int gy = tileStartY + local_y - HALO;
        int gx = tileStartX + local_x - HALO;

// Load all 3 channels
#pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
          sTile[local_y][local_x][c][0] = get_pix_value_clamped_rgb(pred, gy, gx, c, H, W);
          sTile[local_y][local_x][c][1] = get_pix_value_clamped_rgb(gt, gy, gx, c, H, W);
        }
      }
    }
  }
  block.sync();

  // ------------------------------------------------------------
  // 3) Horizontal convolution (11x1) - Per Channel
  // ------------------------------------------------------------
  {
    int ly = threadIdx.y;
    int lx = threadIdx.x + HALO;

    // Loop over channels
    for (int c = 0; c < CHANNELS; ++c) {
      float sumX = 0, sumX2 = 0, sumY = 0, sumY2 = 0, sumXY = 0;

#pragma unroll
      for (int d = 1; d <= HALO; ++d) {
        float w = cGauss[HALO - d];
        float Xl = sTile[ly][lx - d][c][0];
        float Yl = sTile[ly][lx - d][c][1];
        float Xr = sTile[ly][lx + d][c][0];
        float Yr = sTile[ly][lx + d][c][1];
        sumX += (Xl + Xr) * w;
        sumX2 += (Xl * Xl + Xr * Xr) * w;
        sumY += (Yl + Yr) * w;
        sumY2 += (Yl * Yl + Yr * Yr) * w;
        sumXY += (Xl * Yl + Xr * Yr) * w;
      }
      {
        float wc = cGauss[HALO];
        float Xc = sTile[ly][lx][c][0];
        float Yc = sTile[ly][lx][c][1];
        sumX += Xc * wc;
        sumX2 += Xc * Xc * wc;
        sumY += Yc * wc;
        sumY2 += Yc * Yc * wc;
        sumXY += Xc * Yc * wc;
      }

      xconv[ly][threadIdx.x][c][0] = sumX;
      xconv[ly][threadIdx.x][c][1] = sumX2;
      xconv[ly][threadIdx.x][c][2] = sumY;
      xconv[ly][threadIdx.x][c][3] = sumY2;
      xconv[ly][threadIdx.x][c][4] = sumXY;

      // Process second row for vertical support if needed
      int ly2 = ly + BLOCK_Y;
      if (ly2 < CONV_Y) {
        sumX = 0;
        sumX2 = 0;
        sumY = 0;
        sumY2 = 0;
        sumXY = 0;
#pragma unroll
        for (int d = 1; d <= HALO; ++d) {
          float w = cGauss[HALO - d];
          float Xl = sTile[ly2][lx - d][c][0];
          float Yl = sTile[ly2][lx - d][c][1];
          float Xr = sTile[ly2][lx + d][c][0];
          float Yr = sTile[ly2][lx + d][c][1];
          sumX += (Xl + Xr) * w;
          sumX2 += (Xl * Xl + Xr * Xr) * w;
          sumY += (Yl + Yr) * w;
          sumY2 += (Yl * Yl + Yr * Yr) * w;
          sumXY += (Xl * Yl + Xr * Yr) * w;
        }
        {
          float wc = cGauss[HALO];
          float Xc = sTile[ly2][lx][c][0];
          float Yc = sTile[ly2][lx][c][1];
          sumX += Xc * wc;
          sumX2 += Xc * Xc * wc;
          sumY += Yc * wc;
          sumY2 += Yc * Yc * wc;
          sumXY += Xc * Yc * wc;
        }
        xconv[ly2][threadIdx.x][c][0] = sumX;
        xconv[ly2][threadIdx.x][c][1] = sumX2;
        xconv[ly2][threadIdx.x][c][2] = sumY;
        xconv[ly2][threadIdx.x][c][3] = sumY2;
        xconv[ly2][threadIdx.x][c][4] = sumXY;
      }
    }
  }
  block.sync();

  // ------------------------------------------------------------
  // 4) Vertical convolution, SSIM Calc, Loss Reduction (Per Channel)
  // ------------------------------------------------------------
  float pixel_loss_sum = 0.0f;

  if (pix_x < W && pix_y < H) {
    int ly = threadIdx.y + HALO;
    int lx = threadIdx.x;

    for (int c = 0; c < CHANNELS; ++c) {
      float out0 = 0, out1 = 0, out2 = 0, out3 = 0, out4 = 0;

#pragma unroll
      for (int d = 1; d <= HALO; ++d) {
        float w = cGauss[HALO - d];
        float *top = xconv[ly - d][lx][c];
        float *bot = xconv[ly + d][lx][c];
        out0 += (top[0] + bot[0]) * w;
        out1 += (top[1] + bot[1]) * w;
        out2 += (top[2] + bot[2]) * w;
        out3 += (top[3] + bot[3]) * w;
        out4 += (top[4] + bot[4]) * w;
      }
      {
        float wc = cGauss[HALO];
        float *ctr = xconv[ly][lx][c];
        out0 += ctr[0] * wc;
        out1 += ctr[1] * wc;
        out2 += ctr[2] * wc;
        out3 += ctr[3] * wc;
        out4 += ctr[4] * wc;
      }

      // Stats
      float mu1 = out0;
      float mu2 = out2;
      float mu1_sq = mu1 * mu1;
      float mu2_sq = mu2 * mu2;
      float sigma1_sq = out1 - mu1_sq;
      float sigma2_sq = out3 - mu2_sq;
      float sigma12 = out4 - mu1 * mu2;

      // SSIM
      float A = mu1_sq + mu2_sq + C1;
      float B = sigma1_sq + sigma2_sq + C2;
      float C_ = 2.f * mu1 * mu2 + C1;
      float D_ = 2.f * sigma12 + C2;
      float ssim_val = (C_ * D_) / (A * B);

      // L1
      // Access interleaved global memory directly for L1 check
      float pred_val = pred[pix_id * CHANNELS + c];
      float gt_val = gt[pix_id * CHANNELS + c];
      float l1_val = fabsf(pred_val - gt_val);

      // Combined Loss for this channel
      pixel_loss_sum += (1.0f - ssim_weight) * l1_val + ssim_weight * (1.0f - ssim_val);

      // -----------------------------------------------------
      // Compute Partial Derivatives for Backward Pass
      // -----------------------------------------------------
      float d_ssim_dmu1 = ((mu2 * 2.f * D_) / (A * B) - (mu2 * 2.f * C_) / (A * B) -
                           (mu1 * 2.f * C_ * D_) / (A * A * B) + (mu1 * 2.f * C_ * D_) / (A * B * B));
      float d_ssim_dsigma1_sq = (-C_ * D_) / (A * B * B);
      float d_ssim_dsigma12 = (2.f * C_) / (A * B);

      // Store interleaved derivatives
      int out_idx = pix_id * CHANNELS + c;
      dm_dmu1[out_idx] = -ssim_weight * d_ssim_dmu1;
      dm_dsigma1_sq[out_idx] = -ssim_weight * d_ssim_dsigma1_sq;
      dm_dsigma12[out_idx] = -ssim_weight * d_ssim_dsigma12;
    }
  }

  // ------------------------------------------------------------
  // 5) Block Reduction for Total Loss
  // ------------------------------------------------------------
  // Using shared memory reduction
  __shared__ float s_loss_red[BLOCK_Y * BLOCK_X];
  int tid = threadIdx.y * BLOCK_X + threadIdx.x;

  // Sum of R, G, B losses for this pixel
  s_loss_red[tid] = pixel_loss_sum;
  block.sync();

  for (unsigned int s = (BLOCK_X * BLOCK_Y) / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_loss_red[tid] += s_loss_red[tid + s];
    }
    block.sync();
  }

  if (tid == 0) {
    atomicAdd(total_loss_ptr, s_loss_red[0]);
  }
}

// ------------------------------------------
// Backward Kernel: Fused Gradient Assembly (RGB)
// ------------------------------------------
__global__ void fused_loss_backward_kernel(int H, int W, float ssim_weight, const float *__restrict__ pred,
                                           const float *__restrict__ gt, float *__restrict__ image_grad,
                                           const float *__restrict__ dm_dmu1, const float *__restrict__ dm_dsigma1_sq,
                                           const float *__restrict__ dm_dsigma12) {
  auto block = cg::this_thread_block();

  const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
  const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
  const int pix_id = pix_y * W + pix_x;

  // Shared memory for partials
  // [Derivative Type][Y][X][Channel]
  __shared__ float sData[3][SHARED_Y][SHARED_X][CHANNELS];
  // [Y][X][Channel][Derivative Accumulator]
  __shared__ float sScratch[CONV_Y][CONV_X][CHANNELS][3];

  float p1[3] = {0.f, 0.f, 0.f};
  float p2[3] = {0.f, 0.f, 0.f};

  if (pix_x < W && pix_y < H) {
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
      p1[c] = pred[pix_id * CHANNELS + c];
      p2[c] = gt[pix_id * CHANNELS + c];
    }
  }

  // ------------------------------------------------------------
  // 1) Load Derivatives Tile (All Channels)
  // ------------------------------------------------------------
  {
    const int start_y = block.group_index().y * BLOCK_Y;
    const int start_x = block.group_index().x * BLOCK_X;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int totalThreads = BLOCK_X * BLOCK_Y;
    int num_warps = (totalThreads + 31) / 32;

    // Parallel loading of the 3 derivative maps for 3 channels
    for (int row = warp_id; row < SHARED_Y; row += num_warps) {
      int gy = start_y + row - HALO;
      for (int col = lane_id; col < SHARED_X; col += 32) {
        int gx = start_x + col - HALO;

#pragma unroll
        for (int c = 0; c < CHANNELS; ++c) {
          // Use clamped fetch for derivatives map
          float vmu = get_pix_value_clamped_rgb(dm_dmu1, gy, gx, c, H, W);
          float vs1 = get_pix_value_clamped_rgb(dm_dsigma1_sq, gy, gx, c, H, W);
          float vs12 = get_pix_value_clamped_rgb(dm_dsigma12, gy, gx, c, H, W);

          sData[0][row][col][c] = vmu;
          sData[1][row][col][c] = vs1;
          sData[2][row][col][c] = vs12;
        }
      }
    }
  }
  block.sync();

  // ------------------------------------------------------------
  // 2) Horizontal Pass (Per Channel)
  // ------------------------------------------------------------
  {
    int ly = threadIdx.y;
    int lx = threadIdx.x + HALO;

    // Process up to 2 rows per thread if BLOCK_Y < CONV_Y
    for (int pass = 0; pass < 2; ++pass) {
      int yy = ly + pass * BLOCK_Y;
      if (yy < CONV_Y) {

        for (int c = 0; c < CHANNELS; ++c) {
          float accum0 = 0, accum1 = 0, accum2 = 0;
#pragma unroll
          for (int d = 1; d <= HALO; ++d) {
            float w = cGauss[HALO - d];
            accum0 += (sData[0][yy][lx - d][c] + sData[0][yy][lx + d][c]) * w;
            accum1 += (sData[1][yy][lx - d][c] + sData[1][yy][lx + d][c]) * w;
            accum2 += (sData[2][yy][lx - d][c] + sData[2][yy][lx + d][c]) * w;
          }
          {
            float wc = cGauss[HALO];
            accum0 += sData[0][yy][lx][c] * wc;
            accum1 += sData[1][yy][lx][c] * wc;
            accum2 += sData[2][yy][lx][c] * wc;
          }
          sScratch[yy][threadIdx.x][c][0] = accum0;
          sScratch[yy][threadIdx.x][c][1] = accum1;
          sScratch[yy][threadIdx.x][c][2] = accum2;
        }
      }
    }
  }
  block.sync();

  // ------------------------------------------------------------
  // 3) Vertical Pass + L1 Gradient Addition (Per Channel)
  // ------------------------------------------------------------
  if (pix_x < W && pix_y < H) {
    int ly = threadIdx.y + HALO;
    int lx = threadIdx.x;

    for (int c = 0; c < CHANNELS; ++c) {
      float sum0 = 0, sum1 = 0, sum2 = 0;
#pragma unroll
      for (int d = 1; d <= HALO; ++d) {
        float w = cGauss[HALO - d];
        float *top = sScratch[ly - d][lx][c];
        float *bot = sScratch[ly + d][lx][c];
        sum0 += (top[0] + bot[0]) * w;
        sum1 += (top[1] + bot[1]) * w;
        sum2 += (top[2] + bot[2]) * w;
      }
      {
        float wc = cGauss[HALO];
        float *ctr = sScratch[ly][lx][c];
        sum0 += ctr[0] * wc;
        sum1 += ctr[1] * wc;
        sum2 += ctr[2] * wc;
      }

      // 1. SSIM Gradient Component
      float ssim_grad_component = ssim_weight * (sum0 + (2.f * p1[c]) * sum1 + (p2[c]) * sum2);

      // 2. L1 Gradient Component
      float l1_grad_component = (1.0f - ssim_weight) * ((p1[c] > p2[c]) ? 1.0f : -1.0f);

      // Final Gradient
      // Normalizing by pixel count * channels
      const float grad_scale = 1.0f / (float)(H * W);
      float total_grad = (ssim_grad_component + l1_grad_component) * grad_scale;

      image_grad[pix_id * CHANNELS + c] = total_grad;
    }
  }
}

float fused_loss(const float *predicted_data, const float *gt_data, int rows, int cols, const float ssim_weight,
                 float *image_grad, cudaStream_t stream) {
  ASSERT_DEVICE_POINTER(predicted_data);
  ASSERT_DEVICE_POINTER(gt_data);
  ASSERT_DEVICE_POINTER(image_grad);

  // Use BLOCK_X/Y defined above
  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid((cols + BLOCK_X - 1) / BLOCK_X, (rows + BLOCK_Y - 1) / BLOCK_Y);

  size_t img_size = rows * cols * CHANNELS * sizeof(float);
  float *d_mu1, *d_sigma1_sq, *d_sigma12;

  // Allocate temporary buffers for partial derivatives
  cudaMallocAsync(&d_mu1, img_size, stream);
  cudaMallocAsync(&d_sigma1_sq, img_size, stream);
  cudaMallocAsync(&d_sigma12, img_size, stream);

  // Total loss accumulator
  thrust::device_vector<float> d_total_loss(1, 0.0f);
  float *d_loss_ptr = thrust::raw_pointer_cast(d_total_loss.data());

  // Constants
  float C1 = SSIMConstants::C1;
  float C2 = SSIMConstants::C2;

  // 1. Forward Pass: Compute Loss + Partial Derivatives
  fused_loss_forward_kernel<<<grid, block, 0, stream>>>(rows, cols, C1, C2, ssim_weight, predicted_data, gt_data,
                                                        d_loss_ptr, d_mu1, d_sigma1_sq, d_sigma12);

  // 2. Backward Pass: Convolve Partials + Add L1
  fused_loss_backward_kernel<<<grid, block, 0, stream>>>(rows, cols, ssim_weight, predicted_data, gt_data, image_grad,
                                                         d_mu1, d_sigma1_sq, d_sigma12);

  cudaFreeAsync(d_mu1, stream);
  cudaFreeAsync(d_sigma1_sq, stream);
  cudaFreeAsync(d_sigma12, stream);

  // Return average loss
  float total_loss = d_total_loss[0];
  return total_loss / (float)(rows * cols * CHANNELS);
}
