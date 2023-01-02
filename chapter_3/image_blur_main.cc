#include <chrono>

#include "common/test_image.h"
#include "image_blur.h"

int main(int argc, const char* argv[]) {
  auto image = common::GetImage();
  auto blur = std::make_unique<unsigned char[]>(common::ImageHeight() * common::ImageWidth() * 3);

  auto start = std::chrono::high_resolution_clock::now();
  BlurGpu(blur.get(), image.get(), common::ImageWidth(), common::ImageHeight(), 7);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "BlurGpu: " << duration.count() << "us" << std::endl;

  common::WriteImage("/tmp/blur.bin", blur.get(), common::ImageWidth(), common::ImageHeight(), 3);
  std::cout << "Saved blur image to /tmp/blur.bin, view with python3 common/view_image.py /tmp/blur.bin 3" << std::endl;

  return 0;
}