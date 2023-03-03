#include "parallel_sum.h"

__global__ void ParallelSumKernelV1(float* d_X, float* d_S) {
  __shared__ float partial_sum[kSize];
  partial_sum[threadIdx.x] = d_X[blockIdx.x * blockDim.x + threadIdx.x];
  __syncthreads();

  unsigned int t = threadIdx.x;
  for (unsigned int stride = 1U; stride < blockDim.x; stride *= 2) {
    if (t % (2 * stride) == 0) {
      partial_sum[t] += partial_sum[t + stride];
    }
    __syncthreads();
  }

  // copy partial sum back to global memory
  if (threadIdx.x == 0) {
    d_S[blockIdx.x] = partial_sum[0];
  }
}

/**
 * @brief Less divergence than the one above
 *
 * @param d_X
 * @return __global__
 */
__global__ void ParallelSumKernelV2(float* d_X, float* d_S) {
  __shared__ float partial_sum[kSize];
  partial_sum[threadIdx.x] = d_X[blockIdx.x * blockDim.x + threadIdx.x];
  __syncthreads();

  unsigned int t = threadIdx.x;
  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride = stride >> 1) {
    if (t < stride) {
      partial_sum[t] += partial_sum[t + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    d_S[blockIdx.x] = partial_sum[0];
  }
}

float ParallelSumGpuV1(float* h_X, int num) {
  float* d_X;
  float* d_S;
  cudaMalloc((void**)&d_X, sizeof(float) * num);
  cudaMalloc((void**)&d_S, sizeof(float) * kSize);

  cudaMemcpy(d_X, h_X, sizeof(float) * num, cudaMemcpyHostToDevice);

  int n_thread = 256;
  int n_block = std::ceil(num / static_cast<double>(n_thread));
  dim3 dim_grid(n_block, 1, 1);
  dim3 dim_block(n_thread, 1, 1);

  ParallelSumKernelV1<<<dim_grid, dim_block>>>(d_X, d_S);
  ParallelSumKernelV1<<<1, kSize>>>(d_S, d_S);

  float sum;
  cudaMemcpy(&sum, d_S, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_X);
  cudaFree(d_S);

  return sum;
}

float ParallelSumGpuV2(float* h_X, int num) {
  float* d_X;
  float* d_S;
  cudaMalloc((void**)&d_X, sizeof(float) * num);
  cudaMalloc((void**)&d_S, sizeof(float) * kSize);

  cudaMemcpy(d_X, h_X, sizeof(float) * num, cudaMemcpyHostToDevice);

  int n_thread = 256;
  int n_block = std::ceil(num / static_cast<double>(n_thread));
  dim3 dim_grid(n_block, 1, 1);
  dim3 dim_block(n_thread, 1, 1);

  ParallelSumKernelV2<<<dim_grid, dim_block>>>(d_X, d_S);
  ParallelSumKernelV2<<<1, kSize>>>(d_S, d_S);

  float sum;
  cudaMemcpy(&sum, d_S, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_X);
  cudaFree(d_S);

  return sum;
}
