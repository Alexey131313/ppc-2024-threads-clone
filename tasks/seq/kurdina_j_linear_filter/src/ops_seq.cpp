// Copyright 2024 Kurdina Julia
#include "seq/kurdina_j_linear_filter/include/ops_seq.hpp"

#include <algorithm>
#include <thread>

using namespace std::chrono_literals;

uint8_t HorizontalSplitSequential::NewColor(int i, int j) {
  int r = ker_size / 2;
  int res = 0;
  int im_ind_n;
  int im_ind_m;
  for (int k = -r; k < r + 1; k++) {
    for (int q = -r; q < r + 1; q++) {
      if (k + i <= 0)
        im_ind_n = 0;
      else if (k + i >= n)
        im_ind_n = n - 1;
      else
        im_ind_n = k + i;

      if (q + j <= 0)
        im_ind_m = 0;
      else if (q + j >= m)
        im_ind_m = m - 1;
      else
        im_ind_m = q + j;
      res += image[im_ind_n * m + im_ind_m] * gauss_kernel[(k + 1) * ker_size + (q + 1)];
    }
  }
  res = (uint8_t)(std::clamp(res, 0, 255));
  return res;
}

bool HorizontalSplitSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  n = reinterpret_cast<int *>(taskData->inputs[1])[0];
  m = reinterpret_cast<int *>(taskData->inputs[1])[1];
  image.resize(n * m);
  for (int i = 0; i < n * m; i++) {
    input_.push_back(reinterpret_cast<uint8_t *>(taskData->inputs[0])[i]);
    gauss_kernel.push_back(reinterpret_cast<uint8_t *>(taskData->inputs[2])[i]);
  }
  for (int i = 0; i < n * m; i++) {
    image[i] = input_[i];
  }
  return true;
}

bool HorizontalSplitSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == taskData->outputs_count[0] && taskData->inputs_count[2] == 9;
}

bool HorizontalSplitSequential::run() {
  internal_order_test();

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      image[i * m + j] = NewColor(i, j);
    }
  }
  return true;
}

bool HorizontalSplitSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < n * m; i++) {
    reinterpret_cast<uint8_t *>(taskData->outputs[0])[i] = image[i];
  }
  return true;
}
