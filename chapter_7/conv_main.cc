#include <chrono>
#include <cmath>
#include <iostream>

#include "conv.h"

void PrintSample(float* h_P, int size, int samples) {
  std::cout << "Result: ";
  for (int i = 0; i < samples; ++i) {
    std::cout << h_P[i] << " ";
  }

  std::cout << "... ";
  for (int i = 0; i < samples; ++i) {
    std::cout << h_P[size - 1 - i] << " ";
  }

  std::cout << std::endl;
}

int main(int argc, const char* argv[]) {
  // prepare data
  int width = 4096 * 4096 + 16;
  int mask_width = kMaskWidth;
  float* h_N = new float[width];
  float* h_M = new float[mask_width];
  float* h_P = new float[width];
  for (int i = 0; i < width; ++i) {
    h_N[i] = std::sin(static_cast<float>(i));
  }
  for (int i = 0; i < mask_width; ++i) {
    h_M[i] = 1.0F;
  }

  {
    auto t1 = std::chrono::high_resolution_clock::now();
    Conv1dCpu(h_N, h_M, h_P, mask_width, width);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "Conv1dCpu: " << d1.count() << "us" << std::endl;
    PrintSample(h_P, width, 5);
  }

  {
    auto t1 = std::chrono::high_resolution_clock::now();
    Conv1dBasicGpu(h_N, h_M, h_P, mask_width, width);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "Conv1dBasicGpu: " << d1.count() << "us" << std::endl;
    PrintSample(h_P, width, 5);
  }

  {
    auto t1 = std::chrono::high_resolution_clock::now();
    Conv1dConstMemGpu(h_N, h_M, h_P, width);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "Conv1dBasicGpu: " << d1.count() << "us" << std::endl;
    PrintSample(h_P, width, 5);
  }

  {
    auto t1 = std::chrono::high_resolution_clock::now();
    TiledConv1dGpu(h_N, h_M, h_P, width);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "TiledConv1dGpu: " << d1.count() << "us" << std::endl;
    PrintSample(h_P, width, 5);
  }

  {
    auto t1 = std::chrono::high_resolution_clock::now();
    GeneralTiledConv1dGpu(h_N, h_M, h_P, width);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "GeneralTiledConv1dGpu: " << d1.count() << "us" << std::endl;
    PrintSample(h_P, width, 5);
  }

  return 0;
}