#include <chrono>
#include <cmath>
#include <iostream>

/**
 * @brief Blur RGB image
 *
 * @param h_out: row major image in host memory
 * @param h_in: row major image in host memory
 * @param width
 * @param height
 * @param half_patch_size: half patch size
 */
void BlurGpu(unsigned char* h_out, unsigned char* h_in, int width, int height, int half_patch_size);

/**
 * @brief Blur RGB image
 *
 * @param h_out: row major image in host memory
 * @param h_in: row major image in host memory
 * @param width
 * @param height
 * @param half_patch_size: half patch size
 */
void BlurCpu(unsigned char* h_out, unsigned char* h_in, int width, int height, int half_patch_size);
