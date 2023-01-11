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
  int thread_x_per_block = 32;
  int thread_y_per_block = 32;
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
