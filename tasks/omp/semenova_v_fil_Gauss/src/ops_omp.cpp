// Copyright 2024 Semenova Veronika
#include "omp/semenova_v_fil_Gauss/include/ops_omp.hpp"

#include <omp.h>

bool ImageFilGauss::validation() {
  internal_order_test();

  return !taskData->inputs.empty() && !taskData->outputs.empty() && !taskData->inputs_count.empty() &&
         !taskData->outputs_count.empty() && taskData->inputs[0] != nullptr && taskData->outputs[0] != nullptr &&
         taskData->outputs_count[1] == taskData->inputs_count[1] && taskData->outputs_count[0] >= 3 &&
         taskData->outputs_count[1] >= 3;  // the image size <>= size of filter core
}

bool ImageFilGauss::pre_processing() {
  internal_order_test();

  try {
    n = taskData->inputs_count[0];
    m = taskData->inputs_count[1];

    image = reinterpret_cast<int*>(taskData->inputs[0]);
    filteredImage = reinterpret_cast<int*>(taskData->outputs[0]);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        *imageIndex(i, j) = std::max(0, std::min(255, *imageIndex(i, j)));
        *filteredIndex(i, j) = *imageIndex(i, j);
      }
    }

  } catch (...) {
    return false;
  }

  return true;
}

bool ImageFilGauss::run() {
  internal_order_test();
  try {
#pragma omp parallel for schedule(static)
    for (int i = 1; i < n - 1; ++i) {
      for (int j = 1; j < m - 1; ++j) {
        double sum = 0;
        sum = *imageIndex(i - 1, j - 1) * kernel[0][0] + *imageIndex(i - 1, j) * kernel[0][1] +
              *imageIndex(i - 1, j + 1) * kernel[0][2] + *imageIndex(i, j - 1) * kernel[1][0] +
              *imageIndex(i, j) * kernel[1][1] + *imageIndex(i, j + 1) * kernel[1][2] +
              *imageIndex(i + 1, j - 1) * kernel[2][0] + *imageIndex(i + 1, j) * kernel[2][1] +
              *imageIndex(i + 1, j + 1) * kernel[2][2];
        *filteredIndex(i, j) = (int)sum;
      }
    }
  } catch (...) {
    return false;
  }
  return true;
}

bool ImageFilGauss::post_processing() {
  internal_order_test();
  try {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        *filteredIndex(i, j) = std::max(0, std::min(255, *filteredIndex(i, j)));
      }
    }
  } catch (...) {
    return false;
  }
  return true;
}
