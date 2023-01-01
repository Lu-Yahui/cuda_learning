#include "vector_addition.h"

__global__ void VecAddKernel(float* A, float* B, float* C, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

/**
 * @brief CPU version of vector addition
 *
 * @param h_A
 * @param h_B
 * @param h_C
 * @param n
 */
void VecAddCpu(float* h_A, float* h_B, float* h_C, int n) {
  for (int i = 0; i < n; ++i) {
    h_C[i] = h_A[i] + h_B[i];
  }
}

/**
 * @brief GPU version of vector addition
 *
 * @param h_A
 * @param h_B
 * @param h_C
 * @param n
 */
void VecAddGpu(float* h_A, float* h_B, float* h_C, int n) {
  int size = n * sizeof(float);

  // allocate device memory
  float* d_A;
  float* d_B;
  float* d_C;
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);

  // copy host memory to device memory
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // kernel launch
  int block_size = static_cast<int>(std::ceil(n / 256.0));
  VecAddKernel<<<block_size, 256>>>(d_A, d_B, d_C, n);

  // copy result from device memory to host memory
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // release cuda memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
