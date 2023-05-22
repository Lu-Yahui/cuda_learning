#include <cuda_runtime_api.h>

#include "conv.h"

constexpr int kTileSize = 512;

__constant__ float M[kMaskWidth];

void Conv1dCpu(float* h_N, float* h_M, float* h_P, int mask_width, int width) {
  for (int i = 0; i < width; ++i) {
    float value = 0.0F;
    int N_start_point = i - mask_width / 2;
    for (int j = 0; j < mask_width; ++j) {
      if (N_start_point + j >= 0 && N_start_point + j < width) {
        value += h_N[N_start_point + j] * h_M[j];
      }
    }

    h_P[i] = value;
  }
}

/**
 * @brief Basic 1D convolution kernel
 *
 * @param N Original data
 * @param M Mask
 * @param P Result
 * @param mask_width Mask width
 * @param width Data width
 */
__global__ void Conv1dBasicKernel(float* N, float* M, float* P, int mask_width, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= width) {
    return;
  }

  float value = 0.0F;
  int N_start_point = i - mask_width / 2;
  for (int j = 0; j < mask_width; ++j) {
    if (N_start_point + j >= 0 && N_start_point + j < width) {
      value += N[N_start_point + j] * M[j];
    }
  }

  P[i] = value;
}

void Conv1dBasicGpu(float* h_N, float* h_M, float* h_P, int mask_width, int width) {
  float* d_N;
  float* d_M;
  float* d_P;
  cudaMalloc((void**)&d_N, sizeof(float) * width);
  cudaMalloc((void**)&d_M, sizeof(float) * mask_width);
  cudaMalloc((void**)&d_P, sizeof(float) * width);

  cudaMemcpy(d_N, h_N, sizeof(float) * width, cudaMemcpyHostToDevice);
  cudaMemcpy(d_M, h_M, sizeof(float) * mask_width, cudaMemcpyHostToDevice);

  int n_thread = 256;
  int n_block = std::ceil(width / static_cast<double>(n_thread));
  dim3 dim_grid(n_block, 1, 1);
  dim3 dim_block(n_thread, 1, 1);

  Conv1dBasicKernel<<<dim_grid, dim_block>>>(d_N, d_M, d_P, mask_width, width);

  cudaMemcpy(h_P, d_P, sizeof(float) * width, cudaMemcpyDeviceToHost);

  cudaFree(d_N);
  cudaFree(d_M);
  cudaFree(d_P);
}

/**
 * @brief Basic 1D convolution kernel with constant memory and caching
 *
 * @param N Original data
 * @param P Result
 * @param width Data width
 */
__global__ void Conv1dConstMemKernel(float* N, float* P, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= width) {
    return;
  }

  float value = 0.0F;
  int N_start_point = i - kMaskWidth / 2;
  for (int j = 0; j < kMaskWidth; ++j) {
    if (N_start_point + j >= 0 && N_start_point + j < width) {
      value += N[N_start_point + j] * M[j];
    }
  }

  P[i] = value;
}

void Conv1dConstMemGpu(float* h_N, float* h_M, float* h_P, int width) {
  // copy mask
  cudaMemcpyToSymbol(M, h_M, sizeof(float) * kMaskWidth);

  float* d_N;
  float* d_P;
  cudaMalloc((void**)&d_N, sizeof(float) * width);
  cudaMalloc((void**)&d_P, sizeof(float) * width);

  cudaMemcpy(d_N, h_N, sizeof(float) * width, cudaMemcpyHostToDevice);

  int n_thread = kTileSize;
  int n_block = std::ceil(width / static_cast<double>(n_thread));
  dim3 dim_grid(n_block, 1, 1);
  dim3 dim_block(n_thread, 1, 1);

  Conv1dConstMemKernel<<<dim_grid, dim_block>>>(d_N, d_P, width);

  cudaMemcpy(h_P, d_P, sizeof(float) * width, cudaMemcpyDeviceToHost);

  cudaFree(d_N);
  cudaFree(d_P);
}

/**
 * @brief Tiled conv 1d kernel with constant memory caching and tiled caching
 *
 * @param N
 * @param P
 * @param width
 */
__global__ void TiledConv1dKernel(float* N, float* P, int width) {
  __shared__ float N_ds[kTileSize + kMaskWidth - 1];

  // half size of mask
  int n = kMaskWidth / 2;

  // load left halo cells
  int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
  // only last n threads in this block are used to load left halo cells
  if (threadIdx.x >= blockDim.x - n) {
    N_ds[threadIdx.x - (blockDim.x - n)] = halo_index_left < 0 ? 0.0F : N[halo_index_left];
  }

  // load center cells
  N_ds[threadIdx.x + n] = N[blockIdx.x * blockDim.x + threadIdx.x];

  // load right halo cells
  int halo_index_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
  // only first n threads in this block are used to load right halo cells
  if (threadIdx.x < n) {
    N_ds[threadIdx.x + blockDim.x + n] = halo_index_right >= width ? 0.0F : N[halo_index_right];
  }

  // sync all threads to ensure loading is done.
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width) {
    float value = 0.0F;
    for (int j = 0; j < kMaskWidth; ++j) {
      value += N_ds[threadIdx.x + j] * M[j];
    }
    P[i] = value;
  }
}

void TiledConv1dGpu(float* h_N, float* h_M, float* h_P, int width) {
  // copy mask to constant GPU memory
  cudaMemcpyToSymbol(M, h_M, sizeof(float) * kMaskWidth);

  float* d_N;
  float* d_P;
  cudaMalloc((void**)&d_N, sizeof(float) * width);
  cudaMalloc((void**)&d_P, sizeof(float) * width);

  cudaMemcpy(d_N, h_N, sizeof(float) * width, cudaMemcpyHostToDevice);

  int threads_per_block = kTileSize;
  int blocks_per_grid = std::ceil(static_cast<double>(width) / static_cast<double>(threads_per_block));
  TiledConv1dKernel<<<blocks_per_grid, threads_per_block>>>(d_N, d_P, width);

  cudaMemcpy(h_P, d_P, sizeof(float) * width, cudaMemcpyDeviceToHost);

  cudaFree(d_N);
  cudaFree(d_P);
}

/**
 * @brief Simpler but general tiled conv1d kernel, by using L2 cache
 *
 * @param N
 * @param P
 * @param width
 */
__global__ void GeneralTiledConv1dKernel(float* N, float* P, int width) {
  // load tile internal cells into shared memory
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float N_ds[kTileSize];
  N_ds[threadIdx.x] = N[i];
  __syncthreads();

  int this_tile_start_point = blockIdx.x * blockDim.x;
  int next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
  int n_start_point = i - kMaskWidth / 2;
  if (i < width) {
    float value = 0.0F;
    for (int j = 0; j < kMaskWidth; ++j) {
      int index = n_start_point + j;
      // valid cells
      if (index >= 0 && index < width) {
        if (index >= this_tile_start_point && index < next_tile_start_point) {
          // load from shared memory
          value += N_ds[threadIdx.x + j - kMaskWidth / 2] * M[j];
        } else {
          // load from L2 cache or global memory
          value += N[index] * M[j];
        }
      } else {
        // padding 0
      }
    }

    P[i] = value;
  }
}

void GeneralTiledConv1dGpu(float* h_N, float* h_M, float* h_P, int width) {
  // copy mask to constant GPU memory
  cudaMemcpyToSymbol(M, h_M, sizeof(float) * kMaskWidth);

  float* d_N;
  float* d_P;
  cudaMalloc((void**)&d_N, sizeof(float) * width);
  cudaMalloc((void**)&d_P, sizeof(float) * width);

  cudaMemcpy(d_N, h_N, sizeof(float) * width, cudaMemcpyHostToDevice);

  int threads_per_block = kTileSize;
  int blocks_per_grid = std::ceil(static_cast<double>(width) / static_cast<double>(threads_per_block));
  GeneralTiledConv1dKernel<<<blocks_per_grid, threads_per_block>>>(d_N, d_P, width);

  cudaMemcpy(h_P, d_P, sizeof(float) * width, cudaMemcpyDeviceToHost);

  cudaFree(d_N);
  cudaFree(d_P);
}
