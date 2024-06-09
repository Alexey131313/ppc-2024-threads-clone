// Copyright 2024 Shishkina Valeriya
#include "omp/shishkina_v_gauss_block_omp/include/ops_omp.hpp"

#include <cmath>
#include <random>
#include <thread>

std::vector<int> getPicture1(int n, int m, uint8_t min, uint8_t max) {
  int size = n * m;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> distrib(min, max);
  std::vector<int> picture(size);
  for (int i = 0; i < size; i++) {
    picture[i] = static_cast<int>(distrib(gen));
  }
  return picture;
}

bool LinearFilteringGauss::pre_processing() {
  internal_order_test();
  height = reinterpret_cast<int *>(taskData->inputs[0])[0];
  width = reinterpret_cast<int *>(taskData->inputs[0])[1];
  min = reinterpret_cast<int *>(taskData->inputs[2])[0];
  max = reinterpret_cast<int *>(taskData->inputs[2])[1];
  for (int i = 0; i < height * width; i++) {
    input.push_back(reinterpret_cast<int *>(taskData->inputs[1])[i]);
    res.push_back(0);
  }
  return true;
}

bool LinearFilteringGauss::validation() {
  internal_order_test();
  bool flag = false;
  if (taskData->inputs_count[1] == taskData->outputs_count[0]) flag = true;
  return flag;
}

bool LinearFilteringGauss::run() {
  internal_order_test();
  std::vector<int> filteredImage(input.size(), 0);
  int blockSize = std::sqrt(input.size());
  // std::cout << "blockSize: " << blockSize;
  std::vector<int> gaussianKernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  int kernelSize = 3;

#pragma omp parallel for
  for (int i = 0; i < height; i += blockSize) {
    for (int j = 0; j < width; j += blockSize) {
      for (int x = i; x < std::min(i + blockSize, height); ++x) {
        for (int y = j; y < std::min(j + blockSize, width); ++y) {
          int sum = 0;
          for (int m = 0; m < kernelSize; ++m) {
            for (int n = 0; n < kernelSize; ++n) {
              int row = x + m - kernelSize / 2;
              int col = y + n - kernelSize / 2;
              if (row >= 0 && row < height && col >= 0 && col < width) {
#pragma omp critical
                { sum += getPixel(row, col) * gaussianKernel[m * kernelSize + n]; }
              }
            }
          }
          sum = std::min(sum, 255);
#pragma omp critical
          { setPixel(x, y, sum); }
        }
      }
    }
  }

  return true;
}
bool LinearFilteringGauss::post_processing() {
  internal_order_test();
  for (int i = 0; i < height * width; i++) {
    reinterpret_cast<int *>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}
