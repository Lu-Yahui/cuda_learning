#include "matrix_mul.h"

__global__ void SimpleMatrixMulKernel(float* M, float* N, float* P, int width) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < width && col < width) {
    float value = 0.0F;
    for (int k = 0; k < width; ++k) {
      value += M[row * width + k] * N[k * width + col];
    }

    P[row * width + col] = value;
  }
}

/**
 * @brief simplified version of tiled matrix multiplication kernel, assume that width == kTileWidth * N
 *
 * @param d_M
 * @param d_N
 * @param d_P
 * @param width
 */
__global__ void SimpleTiledMatrixMulKernel(float* d_M, float* d_N, float* d_P, int width) {
  // allocate shared memory, which all threads within a block can access
  __shared__ float Mds[kTileWidth][kTileWidth];
  __shared__ float Nds[kTileWidth][kTileWidth];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // kTileWidth = blockDim.x and kTileWidth = blockDim.y
  int row = by * kTileWidth + ty;
  int col = bx * kTileWidth + tx;

  // product value
  float value = 0.0F;
  int num_tiles = width / kTileWidth;
  for (int ph = 0; ph < num_tiles; ++ph) {
    // collaborative loading of d_M and d_N into shared memory
    Mds[ty][tx] = d_M[row * width + ph * kTileWidth + tx];
    Nds[ty][tx] = d_N[(ph * kTileWidth + ty) * width + col];
    __syncthreads();

    for (int k = 0; k < kTileWidth; ++k) {
      value += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }

  d_P[row * width + col] = value;
}

__global__ void TiledMatrixMulKernel(float* d_M, float* d_N, float* d_P, int width) {
  // allocate shared memory
  __shared__ float Mds[kTileWidth][kTileWidth];
  __shared__ float Nds[kTileWidth][kTileWidth];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // kTileWidth = blockDim.x and kTileWidth = blockDim.y
  int row = by * kTileWidth + ty;
  int col = bx * kTileWidth + tx;

  // product value
  float value = 0.0F;
  int num_tiles = std::ceil(width / (float)kTileWidth);
  for (int ph = 0; ph < num_tiles; ++ph) {
    // load d_M, d_N from global memory to shared memory
    if (row < width && (ph * kTileWidth + tx) < width) {
      Mds[ty][tx] = d_M[row * width + ph * kTileWidth + tx];
    }
    if (col < width && (ph * kTileWidth + ty) < width) {
      Nds[ty][tx] = d_N[(ph * kTileWidth + ty) * width + col];
    }
    __syncthreads();

    for (int k = 0; k < kTileWidth; ++k) {
      value += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }

  if (row < width && col < width) {
    d_P[row * width + col] = value;
  }
}

void SimpleMatrixMulGpu(float* M, float* N, float* P, int width) {
  // allocate device memory
  float* d_M;
  float* d_N;
  float* d_P;
  int size = sizeof(float) * width * width;
  cudaMalloc((void**)&d_M, size);
  cudaMalloc((void**)&d_N, size);
  cudaMalloc((void**)&d_P, size);

  // copy data from host memory to device memory
  cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);

  // setup grid dim
  int thread_x_per_block = kTileWidth;
  int thread_y_per_block = kTileWidth;
  int block_x_per_grid = std::ceil(width / static_cast<double>(thread_x_per_block));
  int block_y_per_grid = std::ceil(width / static_cast<double>(thread_y_per_block));
  dim3 dim_grid(block_x_per_grid, block_y_per_grid, 1);
  dim3 dim_block(thread_x_per_block, thread_y_per_block, 1);

  // launch kernel
  SimpleMatrixMulKernel<<<dim_grid, dim_block>>>(d_M, d_N, d_P, width);

  // copy result back to host memory
  cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

  // release cuda memory
  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);
}

void TiledMatrixMulGpu(float* M, float* N, float* P, int width) {
  // allocate device memory
  float* d_M;
  float* d_N;
  float* d_P;
  int size = sizeof(float) * width * width;
  cudaMalloc((void**)&d_M, size);
  cudaMalloc((void**)&d_N, size);
  cudaMalloc((void**)&d_P, size);

  // copy data to device memory
  cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);

  // setup grid dim
  int thread_x_per_block = kTileWidth;
  int thread_y_per_block = kTileWidth;
  int block_x_per_grid = std::ceil(width / static_cast<double>(thread_x_per_block));
  int block_y_per_grid = std::ceil(width / static_cast<double>(thread_y_per_block));
  dim3 dim_grid(block_x_per_grid, block_y_per_grid, 1);
  dim3 dim_block(thread_x_per_block, thread_y_per_block, 1);

  // launch kernel
  TiledMatrixMulKernel<<<dim_grid, dim_block>>>(d_M, d_N, d_P, width);

  // copy result back to host memory
  cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

  // release cuda memory
  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);
}
