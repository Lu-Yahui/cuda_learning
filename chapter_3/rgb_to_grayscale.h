#include <chrono>
#include <cmath>
#include <iostream>

/**
 * @brief GPU version of RGB to gray scale
 *
 * @param out: row major grayscale image in host memory
 * @param in: row major rgb image in host memory
 * @param width
 * @param height
 */
void RgbToGrayScaleGpu(unsigned char* out, unsigned char* in, int width, int height);
