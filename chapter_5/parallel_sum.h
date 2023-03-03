#include <chrono>
#include <cmath>
#include <iostream>

constexpr int kSize = 256;

float ParallelSumGpuV1(float* h_X, int num);

float ParallelSumGpuV2(float* h_X, int num);
