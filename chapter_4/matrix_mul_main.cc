#include <chrono>
#include <iostream>

#include "matrix_mul.h"

void MakeSquareMatrix(float* M, float* N, int width) {
  // Identity
  for (int r = 0; r < width; ++r) {
    for (int c = 0; c < width; ++c) {
      int index = r * width + c;
      if (r == c) {
        M[index] = 1.0F;
        N[index] = 1.0F;
      } else {
        M[index] = 0.0F;
        N[index] = 0.0F;
      }
    }
  }
}

int main(int argc, const char* argv[]) {
  int width = 8192;
  float* M = new float[width * width];
  float* N = new float[width * width];
  float* P = new float[width * width];

  MakeSquareMatrix(M, N, width);

  {
    auto start = std::chrono::high_resolution_clock::now();
    SimpleMatrixMulGpu(M, N, P, width);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "SimpleMatrixMulGpu: " << duration.count() << "us." << std::endl;
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    TiledMatrixMulGpu(M, N, P, width);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "SimpleMatrixMulGpu: " << duration.count() << "us." << std::endl;
  }

  delete[] M;
  delete[] N;
  delete[] P;

  return 0;
}