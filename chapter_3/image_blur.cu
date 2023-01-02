#include "image_blur.h"

/**
 * @brief Blur kernel
 *
 * @param out: row major image in host memory
 * @param in: row major image in host memory
 * @param width
 * @param height
 * @param half_patch_size: half patch size, e.g. 1 for 3x3 patch
 * @return __global__
 */
__global__ void BlurKernel(unsigned char* out, unsigned char* in, int width, int height, int half_patch_size) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (col < width && row < height) {
    int pixel_val_r = 0;
    int pixel_val_g = 0;
    int pixel_val_b = 0;
    int pixel_num = 0;
    for (int blur_row = -half_patch_size; blur_row < half_patch_size + 1; ++blur_row) {
      for (int blur_col = -half_patch_size; blur_col < half_patch_size + 1; ++blur_col) {
        int curr_row = row + blur_row;
        int curr_col = col + blur_col;
        if (curr_row > -1 && curr_row < height && curr_col > -1 && curr_col < width) {
          int offset = 3 * (curr_row * width + curr_col);
          pixel_val_r += in[offset];
          pixel_val_g += in[offset + 1];
          pixel_val_b += in[offset + 2];
          pixel_num++;
        }
      }
    }

    // write back to out
    int out_offset = 3 * (row * width + col);
    out[out_offset] = (unsigned char)(pixel_val_r / pixel_num);
    out[out_offset + 1] = (unsigned char)(pixel_val_g / pixel_num);
    out[out_offset + 2] = (unsigned char)(pixel_val_b / pixel_num);
  }
}

void BlurGpu(unsigned char* h_out, unsigned char* h_in, int width, int height, int half_patch_size) {
  // allocate device memory
  unsigned char* d_out;
  unsigned char* d_in;
  int size = sizeof(unsigned char) * width * height * 3;
  cudaMalloc((void**)&d_out, size);
  cudaMalloc((void**)&d_in, size);

  // copy original image from host to device
  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  // launch kernel
  int thread_x_per_block = 32;
  int thread_y_per_block = 32;
  int block_x_per_grid = std::ceil(width / static_cast<double>(thread_x_per_block));
  int block_y_per_grid = std::ceil(height / static_cast<double>(thread_y_per_block));
  dim3 dim_grid(block_x_per_grid, block_y_per_grid, 1);
  dim3 dim_block(thread_x_per_block, thread_y_per_block, 1);
  BlurKernel<<<dim_grid, dim_block>>>(d_out, d_in, width, height, half_patch_size);

  // copy result back to host memory
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  // release device memory
  cudaFree(d_out);
  cudaFree(d_in);
}
