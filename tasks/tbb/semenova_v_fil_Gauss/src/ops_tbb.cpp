// Copyright 2024 Semenova Veronika
#include "tbb/semenova_v_fil_Gauss/include/ops_tbb.hpp"

#include <tbb/tbb.h>

int Min(int b, int c) { return (b < c) ? b : c; }

int Max(int a, int b) { return (a > b) ? a : b; }

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

    tbb::parallel_for(tbb::blocked_range2d<int>(0, n, 0, m), [&](const tbb::blocked_range2d<int>& r) {
      for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
        for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
          *imageIndex(i, j) = Max(0, Min(255, *imageIndex(i, j)));
          *filteredIndex(i, j) = *imageIndex(i, j);
        }
      }
    });
  } catch (...) {
    return false;
  }

  return true;
}

/*
bool ImageFilGauss::pre_processing() {
  internal_order_test();

  try {
    n = taskData->inputs_count[0];
    m = taskData->inputs_count[1];

    image = reinterpret_cast<int*>(taskData->inputs[0]);
    filteredImage = reinterpret_cast<int*>(taskData->outputs[0]);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; j++) {
        *imageIndex(i,j) = Max(0, Min(255, *imageIndex(i,j)));
        *filteredIndex(i,j) = *imageIndex(i,j);
      }
    }
  } catch (...) {
    return false;
  }

  return true;
} */

bool ImageFilGauss::run() {
  internal_order_test();
  try {
    tbb::parallel_for(tbb::blocked_range2d<int>(1, n - 1, 1, m - 1), [&](const tbb::blocked_range2d<int>& r) {
      for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
        for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
          double sum = 0;
          sum = *imageIndex(i - 1, j - 1) * kernel[0][0] + *imageIndex(i - 1, j) * kernel[0][1] +
                *imageIndex(i - 1, j + 1) * kernel[0][2] + *imageIndex(i, j - 1) * kernel[1][0] +
                *imageIndex(i, j) * kernel[1][1] + *imageIndex(i, j + 1) * kernel[1][2] +
                *imageIndex(i + 1, j - 1) * kernel[2][0] + *imageIndex(i + 1, j) * kernel[2][1] +
                *imageIndex(i + 1, j + 1) * kernel[2][2];
          *filteredIndex(i, j) = (int)sum;
        }
      }
    });

  } catch (...) {
    return false;
  }
  return true;
}

/*
bool ImageFilGauss::post_processing() {
  internal_order_test();
  try {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        *filteredIndex(i,j) = Max(0, Min(255, *filteredIndex(i,j)));
      }
    }
  }
   catch (...) {
    return false;
  }
  return true;
}*/

bool ImageFilGauss::post_processing() {
  internal_order_test();
  try {
    tbb::parallel_for(tbb::blocked_range2d<int>(0, n, 0, m), [&](const tbb::blocked_range2d<int>& r) {
      for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
        for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
          *filteredIndex(i, j) = Max(0, Min(255, *filteredIndex(i, j)));
        }
      }
    });
  } catch (...) {
    return false;
  }
  return true;
}
