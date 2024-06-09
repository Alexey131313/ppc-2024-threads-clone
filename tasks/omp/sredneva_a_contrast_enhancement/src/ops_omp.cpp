// Copyright 2023 Sredneva Anastasiya
#include "omp/sredneva_a_contrast_enhancement/include/ops_omp.hpp"

#include <omp.h>

#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<uint8_t> getRandomPicture(int n, int m, uint8_t min, uint8_t max) {
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

bool ContrastEnhancement_OMP_Sequential::pre_processing() {
  internal_order_test();
  n = reinterpret_cast<int *>(taskData->inputs[0])[0];
  m = reinterpret_cast<int *>(taskData->inputs[0])[1];
  min = reinterpret_cast<uint8_t *>(taskData->inputs[2])[0];
  max = reinterpret_cast<uint8_t *>(taskData->inputs[2])[1];
  for (int i = 0; i < n * m; i++) {
    input_.push_back(reinterpret_cast<uint8_t *>(taskData->inputs[1])[i]);
    res.push_back(0);
  }
  return true;
}

bool ContrastEnhancement_OMP_Sequential::validation() {
  internal_order_test();
  return taskData->inputs_count[1] == taskData->outputs_count[0];
}

bool ContrastEnhancement_OMP_Sequential::run() {
  internal_order_test();
  int size = n * m;
  if ((size == 0) || (min == max)) {
    return false;
  }
  for (int i = 0; i < size; i++) {
    res[i] = (input_[i] - min) * 255 / (max - min);
  }
  return true;
}

bool ContrastEnhancement_OMP_Sequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < n * m; i++) {
    reinterpret_cast<uint8_t *>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}

bool ContrastEnhancement_OMP_Parallel::pre_processing() {
  internal_order_test();
  n = reinterpret_cast<int *>(taskData->inputs[0])[0];
  m = reinterpret_cast<int *>(taskData->inputs[0])[1];
  min = reinterpret_cast<uint8_t *>(taskData->inputs[2])[0];
  max = reinterpret_cast<uint8_t *>(taskData->inputs[2])[1];
  for (int i = 0; i < n * m; i++) {
    input_.push_back(reinterpret_cast<uint8_t *>(taskData->inputs[1])[i]);
    res.push_back(0);
  }
  return true;
}

bool ContrastEnhancement_OMP_Parallel::validation() {
  internal_order_test();
  return taskData->inputs_count[1] == taskData->outputs_count[0];
}

bool ContrastEnhancement_OMP_Parallel::run() {
  internal_order_test();
  int size = n * m;
  std::vector<uint8_t> temp_res(size, 0);
  if ((size == 0) || (min == max)) {
    return false;
  }
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    temp_res[i] = (input_[i] - min) * 255 / (max - min);
  }
  res = temp_res;
  return true;
}

bool ContrastEnhancement_OMP_Parallel::post_processing() {
  internal_order_test();
  for (int i = 0; i < n * m; i++) {
    reinterpret_cast<uint8_t *>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}
