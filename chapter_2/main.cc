#include "vector_addition.h"

bool CheckResult(float* result, int n) {
  for (int i = 0; i < n; ++i) {
    if (std::abs(result[i] - 3.0F) > 1E-6) {
      return false;
    }
  }

  return true;
}

int main(int argc, const char* argv[]) {
  constexpr int n = 999999999;
  float* h_A = new float[n];
  float* h_B = new float[n];
  float* h_C = new float[n];
  for (int i = 0; i < n; ++i) {
    h_A[i] = 1.0F;
    h_B[i] = 2.0F;
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    VecAddCpu(h_A, h_B, h_C, n);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "VecAddCPU: " << duration.count() << "us, result's good: " << CheckResult(h_C, n) << std::endl;
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    VecAddGpu(h_A, h_B, h_C, n);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "VecAddCPU: " << duration.count() << "us, result's good: " << CheckResult(h_C, n) << std::endl;
  }

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}