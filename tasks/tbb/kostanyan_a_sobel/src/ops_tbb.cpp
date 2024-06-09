// Copyright Kostanyan Arsen 2024
#include "tbb/kostanyan_a_sobel/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <random>
#include <vector>
#undef min

namespace kostanyan_tbb_sobel {

std::vector<uint8_t> kostanyan_getRandomPicture(int n, int m, uint8_t min, uint8_t max) {
  int size = n * m;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> distrib(min, max);
  std::vector<uint8_t> pict(size);
  for (int i = 0; i < size; i++) {
    pict[i] = distrib(gen);
  }
  return pict;
}

bool Kostanyan_EdgeDetectionTBBSequential::pre_processing() {
  internal_order_test();
  n = reinterpret_cast<int *>(taskData->inputs[0])[0];
  m = reinterpret_cast<int *>(taskData->inputs[0])[1];
  for (int i = 0; i < n * m; i++) {
    input_.push_back(reinterpret_cast<uint8_t *>(taskData->inputs[1])[i]);
    res.push_back(0);
  }
  return true;
}

bool Kostanyan_EdgeDetectionTBBSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[1] == taskData->outputs_count[0];
}

bool Kostanyan_EdgeDetectionTBBSequential::run() {
  internal_order_test();
  int size = n * m;
  if (size == 0) {
    return false;
  }
  if (size == 1) {
    res[0] = input_[0];
    return true;
  }

  for (int i = 1; i < n - 1; i++) {
    for (int j = 1; j < m - 1; j++) {
      int gx = -input_[(i - 1) * m + j - 1] + input_[(i - 1) * m + j + 1] - 2 * input_[i * m + j - 1] +
               2 * input_[i * m + j + 1] - input_[(i + 1) * m + j - 1] + input_[(i + 1) * m + j + 1];
      int gy = input_[(i - 1) * m + j - 1] + 2 * input_[(i - 1) * m + j] + input_[(i - 1) * m + j + 1] -
               input_[(i + 1) * m + j - 1] - 2 * input_[(i + 1) * m + j] - input_[(i + 1) * m + j + 1];
      res[i * m + j] = sqrt(gx * gx + gy * gy);
    }
  }
  return true;
}

bool Kostanyan_EdgeDetectionTBBSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < n * m; i++) {
    reinterpret_cast<uint8_t *>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}

bool Kostanyan_EdgeDetectionTBBParallel::pre_processing() {
  internal_order_test();
  n = reinterpret_cast<int *>(taskData->inputs[0])[0];
  m = reinterpret_cast<int *>(taskData->inputs[0])[1];
  for (int i = 0; i < n * m; i++) {
    input_.push_back(reinterpret_cast<uint8_t *>(taskData->inputs[1])[i]);
    res.push_back(0);
  }
  return true;
}

bool Kostanyan_EdgeDetectionTBBParallel::validation() {
  internal_order_test();
  return taskData->inputs_count[1] == taskData->outputs_count[0];
}

bool Kostanyan_EdgeDetectionTBBParallel::run() {
  internal_order_test();
  int size = n * m;
  if (size == 0) {
    return false;
  }
  if (size == 1) {
    res[0] = input_[0];
    return true;
  }

  tbb::parallel_for(tbb::blocked_range<int>(1, n - 1), [&](const tbb::blocked_range<int> &range) {
    for (int i = range.begin(); i != range.end(); ++i) {
      for (int j = 1; j < m - 1; j++) {
        int gx = -input_[(i - 1) * m + j - 1] + input_[(i - 1) * m + j + 1] - 2 * input_[i * m + j - 1] +
                 2 * input_[i * m + j + 1] - input_[(i + 1) * m + j - 1] + input_[(i + 1) * m + j + 1];
        int gy = input_[(i - 1) * m + j - 1] + 2 * input_[(i - 1) * m + j] + input_[(i - 1) * m + j + 1] -
                 input_[(i + 1) * m + j - 1] - 2 * input_[(i + 1) * m + j] - input_[(i + 1) * m + j + 1];
        res[i * m + j] = sqrt(gx * gx + gy * gy);
      }
    }
  });

  return true;
}

bool Kostanyan_EdgeDetectionTBBParallel::post_processing() {
  internal_order_test();
  for (int i = 0; i < n * m; i++) {
    reinterpret_cast<uint8_t *>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}
}  // namespace kostanyan_tbb_sobel