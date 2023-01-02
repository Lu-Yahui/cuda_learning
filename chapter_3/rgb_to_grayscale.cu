#include "rgb_to_grayscale.h"

/**
 * @brief RGB to gray scale conversion kernel
 * L = r * 0.21 + g * 0.72 + b * 0.07
 *
 * @param out: row major image output
 * @param in: row major image input
 * @param width
 * @param height
 * @return __global__
 */
__global__ void RgbToGrayScaleKernel(unsigned char* out, unsigned char* in, int width, int height) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  if (col < width && row < height) {
    int gray_offset = row * width + col;
    // 3 channel for RGB per pixel
    int rgb_offset = gray_offset * 3;
    unsigned char r = in[rgb_offset];
    unsigned char g = in[rgb_offset + 1];
    unsigned char b = in[rgb_offset + 2];
    out[gray_offset] = r * 0.21 + g * 0.72 + b * 0.07;
  }
}

void RgbToGrayScaleGpu(unsigned char* out, unsigned char* in, int width, int height) {
  // gray scale
  int size_out = sizeof(unsigned char) * width * height;
  // rgb
  int size_in = sizeof(unsigned char) * width * height * 3;

  // allocate device memory
  unsigned char* d_out;
  unsigned char* d_in;
  cudaMalloc((void**)&d_out, size_out);
  cudaMalloc((void**)&d_in, size_in);

  // copy rgb image from host to device memory
  cudaMemcpy(d_in, in, size_in, cudaMemcpyHostToDevice);

  // launch kernel
  int thread_dim_x_per_block = 32;
  int thread_dim_y_per_block = 32;
  int block_size_x = std::ceil(width / static_cast<double>(thread_dim_x_per_block));
  int block_size_y = std::ceil(height / static_cast<double>(thread_dim_y_per_block));
  dim3 dim_grid(block_size_x, block_size_y, 1);
  dim3 dim_block(thread_dim_x_per_block, thread_dim_y_per_block, 1);
  RgbToGrayScaleKernel<<<dim_grid, dim_block>>>(d_out, d_in, width, height);

  // copy result back to host memory
  cudaMemcpy(out, d_out, size_out, cudaMemcpyDeviceToHost);

  // release device memory
  cudaFree(d_out);
  cudaFree(d_in);
}
