// Copyright 2024 Semenova Veronika
#include "stl/semenova_v_fil_Gauss/include/ops_stl.hpp"

#include <thread>

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

    int numThreads = std::thread::hardware_concurrency();
    auto* threads = new std::thread[numThreads];
    int rowsPerThread = n / numThreads;

    for (int i = 0; i < numThreads; ++i) {
      int startRow = i * rowsPerThread;
      int endRow = (i == numThreads - 1) ? n : (i + 1) * rowsPerThread;
      threads[i] = std::thread([this, startRow, endRow]() {
        for (int i = startRow; i < endRow; ++i) {
          for (int j = 0; j < m; ++j) {
            *imageIndex(i, j) = std::max(0, std::min(255, *imageIndex(i, j)));
            *filteredIndex(i, j) = *imageIndex(i, j);
          }
        }
      });
    }

    for (int i = 0; i < numThreads; ++i) {
      threads[i].join();
    }

    delete[] threads;
  } catch (...) {
    return false;
  }

  return true;
}

bool ImageFilGauss::run() {
  internal_order_test();
  try {
    int numThreads = std::thread::hardware_concurrency();
    auto* threads = new std::thread[numThreads];
    int rowsPerThread = (n - 2) / numThreads;

    for (int i = 0; i < numThreads; ++i) {
      int startRow = i * rowsPerThread + 1;
      int endRow = (i == numThreads - 1) ? n - 1 : (i + 1) * rowsPerThread + 1;
      threads[i] = std::thread([this, startRow, endRow]() {
        for (int i = startRow; i < endRow; ++i) {
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
      });
    }

    for (int i = 0; i < numThreads; ++i) {
      threads[i].join();
    }

    delete[] threads;
  } catch (...) {
    return false;
  }
  return true;
}

bool ImageFilGauss::post_processing() {
  internal_order_test();
  try {
    int numThreads = std::thread::hardware_concurrency();
    auto* threads = new std::thread[numThreads];
    int rowsPerThread = n / numThreads;

    for (int i = 0; i < numThreads; ++i) {
      int startRow = i * rowsPerThread;
      int endRow = (i == numThreads - 1) ? n : (i + 1) * rowsPerThread;
      threads[i] = std::thread([this, startRow, endRow]() {
        for (int i = startRow; i < endRow; ++i) {
          for (int j = 0; j < m; ++j) {
            *filteredIndex(i, j) = std::max(0, std::min(255, *filteredIndex(i, j)));
          }
        }
      });
    }

    for (int i = 0; i < numThreads; ++i) {
      threads[i].join();
    }

    delete[] threads;
  } catch (...) {
    return false;
  }
  return true;
}
