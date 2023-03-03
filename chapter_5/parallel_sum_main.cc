#include <chrono>
#include <iostream>
#include <memory>

#include "parallel_sum.h"

int main(int argc, const char* argv[]) {
  int num = 65536;
  auto h_X = std::make_unique<float[]>(num);
  for (int i = 0; i < num; ++i) {
    h_X[i] = 1.0F;
  }

  {
    auto t1 = std::chrono::high_resolution_clock::now();
    float sum = ParallelSumGpuV1(h_X.get(), num);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "Sum: " << sum << ", elapsed: " << elapsed.count() << std::endl;
  }

  {
    auto t1 = std::chrono::high_resolution_clock::now();
    float sum = ParallelSumGpuV2(h_X.get(), num);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "Sum: " << sum << ", elapsed: " << elapsed.count() << std::endl;
  }

  return 0;
}
