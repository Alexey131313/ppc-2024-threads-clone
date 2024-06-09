// Copyright 2024 Lysanova Julia
#include "seq/lysanova_j_matr_fox/include/ops_seq.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

bool LysanovaTaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs[0] != nullptr) && (taskData->inputs[1] != nullptr) && (taskData->outputs[0] != nullptr) &&
         taskData->inputs_count[0] == taskData->inputs_count[1] &&
         taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool LysanovaTaskSequential::pre_processing() {
  internal_order_test();
  A = reinterpret_cast<double*>(taskData->inputs[0]);
  B = reinterpret_cast<double*>(taskData->inputs[1]);
  C = reinterpret_cast<double*>(taskData->outputs[0]);
  n = round(sqrt(taskData->inputs_count[0]));
  memset(C, 0, n * n * sizeof(*C));
  return true;
}

bool LysanovaTaskSequential::run() {
  internal_order_test();
  try {
    double c = 0;
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) {
        c = 0;
        for (size_t k = 0; k < n; k++) {
          c += A[i * n + k] * B[k * n + j];
        }
        C[i * n + j] = c;
      }
    }
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

bool LysanovaTaskSequential::post_processing() {
  internal_order_test();
  return true;
}

void RandomFillMatrix(double* matrix, int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < sz; i++) {
    matrix[i] = gen() % 100;
  }
}

void EMatrix(double* matrix, int n, double k) {
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      matrix[i * n + j] = 0;
    }
    matrix[i * n + i] = k;
  }
}
void EsdMatrix(double* matrix, int n, double k) {
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      matrix[i * n + (n - i - 1)] = 0;
    }
    matrix[i * n + (n - i - 1)] = k;
  }
}
