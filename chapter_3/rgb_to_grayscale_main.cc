#include <chrono>

#include "common/test_image.h"
#include "rgb_to_grayscale.h"

int main(int argc, const char* argv[]) {
  auto image = common::GetImage();
  auto grayscale = std::make_unique<unsigned char[]>(common::ImageHeight() * common::ImageWidth() * 1);

  auto start = std::chrono::high_resolution_clock::now();
  RgbToGrayScaleGpu(grayscale.get(), image.get(), common::ImageWidth(), common::ImageHeight());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "RGB to grayscale: " << duration.count() << "us" << std::endl;

  common::WriteImage("/tmp/grayscale.bin", grayscale.get(), common::ImageWidth(), common::ImageHeight(), 1);
  std::cout << "Saved grayscale image to /tmp/grayscale.bin, view with python3 common/view_image.py /tmp/grayscale.bin"
            << std::endl;

  return 0;
}