#include <chrono>
#include <cmath>
#include <iostream>

constexpr int kMaskWidth{9};

void Conv1dCpu(float* h_N, float* h_M, float* h_P, int mask_width, int width);

void Conv1dBasicGpu(float* h_N, float* h_M, float* h_P, int mask_width, int width);

void Conv1dConstMemGpu(float* h_N, float* h_M, float* h_P, int width);

void TiledConv1dGpu(float* h_N, float* h_M, float* h_P, int width);

void GeneralTiledConv1dGpu(float* h_N, float* h_M, float* h_P, int width);
