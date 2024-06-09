// Copyright 2024 Ulyanov Daniil
#include "omp/ulyanov_daniil/include/ops_omp.hpp"

#include <omp.h>

#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

// using namespace std::chrono_literals;

void createKernelUlyanov(std::vector<float>& kernel, float sigma) {
  float norm = 0.0;
  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      kernel[(i + 1) * 3 + (j + 1)] = expf(-static_cast<float>(i * i + j * j) / (2 * sigma * sigma));
      norm += kernel[(i + 1) * 3 + (j + 1)];
    }
  }
  for (int i = 0; i < 9; i++) {
    kernel[i] = kernel[i] / norm;
  }
}

bool OMPFilterGaussHorizontalSeqUlyanov::pre_processing() {
  internal_order_test();

  height = taskData->inputs_count[0];
  width = taskData->inputs_count[1];

  uint8_t* data = taskData->inputs[0];

  inputImage = std::vector<uint8_t>(height * width * 3);
  resultImage = std::vector<uint8_t>(height * width * 3);

  kernel = std::vector<float>(9);
  createKernelUlyanov(kernel, 2.0);

  for (int i = 0; i < height * width * 3; i++) {
    inputImage[i] = data[i];
  }

  return true;
}

bool OMPFilterGaussHorizontalSeqUlyanov::validation() {
  internal_order_test();

  return taskData->inputs[0] != nullptr && taskData->outputs[0] != nullptr && taskData->inputs_count[0] != 0 &&
         taskData->inputs_count[1] != 0 && taskData->outputs_count[0] != 0 && taskData->outputs_count[1] != 0;
}

uint8_t calcColorUlyanov(std::vector<uint8_t>& image, int height, int i, int j, std::vector<float>& kernel,
                         int numColor) {
  float color = 0.0;

  for (int l = -1; l < 2; l++) {
    for (int k = -1; k < 2; k++) {
      color += image[(((i + l) * height) + (j + k)) * 3 + numColor] * kernel[((l + 1) * 3) + (k + 1)];
    }
  }
  return static_cast<uint8_t>(color);
}

bool OMPFilterGaussHorizontalSeqUlyanov::run() {
  internal_order_test();
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      for (int l = 0; l < 3; l++) {
        if (i == 0 || i == height - 1 || j == 0 || j == width - 1) {
          resultImage[(i * height + j) * 3 + l] = inputImage[(i * height + j) * 3 + l];
        } else {
          resultImage[(i * height + j) * 3 + l] = calcColorUlyanov(inputImage, height, i, j, kernel, l);
        }
      }
    }
  }
  // std::this_thread::sleep_for(20ms);
  return true;
}

bool OMPFilterGaussHorizontalSeqUlyanov::post_processing() {
  internal_order_test();

  for (int i = 0; i < height * width * 3; i++) {
    taskData->outputs[0][i] = resultImage[i];
  }

  return true;
}

bool OMPFilterGaussHorizontalParUlyanov::pre_processing() {
  internal_order_test();

  height = taskData->inputs_count[0];
  width = taskData->inputs_count[1];

  uint8_t* data = taskData->inputs[0];

  inputImage = std::vector<uint8_t>(height * width * 3);
  resultImage = std::vector<uint8_t>(height * width * 3);

  kernel = std::vector<float>(9);
  createKernelUlyanov(kernel, 2.0);

  for (int i = 0; i < height * width * 3; i++) {
    inputImage[i] = data[i];
  }

  return true;
}

bool OMPFilterGaussHorizontalParUlyanov::validation() {
  internal_order_test();

  return taskData->inputs[0] != nullptr && taskData->outputs[0] != nullptr && taskData->inputs_count[0] != 0 &&
         taskData->inputs_count[1] != 0 && taskData->outputs_count[0] != 0 && taskData->outputs_count[1] != 0;
}

bool OMPFilterGaussHorizontalParUlyanov::run() {
  internal_order_test();
  double start = omp_get_wtime();

#pragma omp parallel for schedule(static)
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      for (int l = 0; l < 3; l++) {
        if (i == 0 || i == height - 1 || j == 0 || j == width - 1) {
          resultImage[(i * height + j) * 3 + l] = inputImage[(i * height + j) * 3 + l];
        } else {
          resultImage[(i * height + j) * 3 + l] = calcColorUlyanov(inputImage, height, i, j, kernel, l);
        }
      }
    }
  }

  double finish = omp_get_wtime();
  std::cout << "How measure time in OpenMP: " << finish - start << std::endl;
  return true;
}

bool OMPFilterGaussHorizontalParUlyanov::post_processing() {
  internal_order_test();

  for (int i = 0; i < height * width * 3; i++) {
    taskData->outputs[0][i] = resultImage[i];
  }

  return true;
}
