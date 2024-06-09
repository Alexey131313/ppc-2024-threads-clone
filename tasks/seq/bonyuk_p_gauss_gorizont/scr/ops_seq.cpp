// Copyright 2024 Bonyuk Peter
#include "seq/bonyuk_p_gauss_gorizont/include/ops_seq.hpp"

#include <cmath>
#include <random>
#include <thread>

using namespace std::chrono_literals;

std::vector<int> getImage(int n, int m, uint8_t min, uint8_t max) {
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

bool LinearGaussianFiltering::pre_processing() {
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

bool LinearGaussianFiltering::validation() {
  internal_order_test();
  bool flag = false;
  if (taskData->inputs_count[1] == taskData->outputs_count[0]) flag = true;
  return flag;
}

bool LinearGaussianFiltering::run() {
  internal_order_test();
  std::vector<int> filteredImage(input.size(), 0);
  std::vector<int> gauss_kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  int kSize = 3;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int sum = 0;
      for (int m = 0; m < kSize; ++m) {
        for (int n = 0; n < kSize; ++n) {
          int row = i + m - kSize / 2;
          int col = j + n - kSize / 2;
          if (row >= 0 && row < height && col >= 0 && col < width) {
            sum += getPixel(row, col) * gauss_kernel[m * kSize + n];
          }
        }
      }
      sum = std::min(sum, 255);
      setPixel(i, j, sum);
    }
  }
  return true;
}

bool LinearGaussianFiltering::post_processing() {
  internal_order_test();
  for (int i = 0; i < height * width; i++) {
    reinterpret_cast<int *>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}
