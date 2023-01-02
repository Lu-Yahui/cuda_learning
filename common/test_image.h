#ifndef COMMON_TEST_IMAGE_H
#define COMMON_TEST_IMAGE_H

#include <fstream>
#include <memory>
#include <string>

namespace common {

inline int ImageWidth() {
  return 1443;
}

inline int ImageHeight() {
  return 1080;
}

inline int ImageChannel() {
  return 3;
}

inline std::unique_ptr<unsigned char[]> GetImage() {
  constexpr char kImageFile[] = "common/cats.image.bin";
  std::unique_ptr<unsigned char[]> image =
      std::make_unique<unsigned char[]>(ImageWidth() * ImageHeight() * ImageChannel());
  std::ifstream ifs(kImageFile);
  ifs.read(reinterpret_cast<char*>(image.get()), sizeof(unsigned char) * ImageWidth() * ImageHeight() * ImageChannel());
  return image;
}

inline void WriteImage(const std::string& filename, unsigned char* image, int width, int height, int channel) {
  std::ofstream ofs(filename);
  ofs.write(reinterpret_cast<char*>(image), sizeof(unsigned char) * width * height * channel);
}

}  // namespace common

#endif  // COMMON_TEST_IMAGE_H
