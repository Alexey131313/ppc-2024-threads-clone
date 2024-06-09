// Copyright 2024 Safronov Mikhail
#include "seq/safronov_m/include/ops_seq.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

using namespace std::chrono_literals;

bool SafronovSeqFoxAlgTask::validation() {
  internal_order_test();
  return (taskData->inputs[0] != nullptr) && (taskData->inputs[1] != nullptr) && (taskData->outputs[0] != nullptr) &&
         taskData->inputs_count[0] == taskData->inputs_count[1] &&
         taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool SafronovSeqFoxAlgTask::pre_processing() {
  internal_order_test();
  A = reinterpret_cast<double *>(taskData->inputs[0]);
  B = reinterpret_cast<double *>(taskData->inputs[1]);
  C = reinterpret_cast<double *>(taskData->outputs[0]);
  n = round(sqrt(taskData->inputs_count[0]));
  memset(C, 0, n * n * sizeof(*C));
  return true;
}

bool SafronovSeqFoxAlgTask::run() {
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
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

bool SafronovSeqFoxAlgTask::post_processing() {
  internal_order_test();

  return true;
}

void GetRandomValue(double *m, int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < size; i++) {
    m[i] = gen() % 100;
  }
}

void identityMatrix(double *m, int n, double k) {
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      m[i * n + j] = 0;
    }
    m[i * n + i] = k;
  }
}

void ModifidentityMatrix(double *m, int n, double k) {
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      m[i * n + (n - i - 1)] = 0;
    }
    m[i * n + (n - i - 1)] = k;
  }
}

std::vector<double> mulSafronov(const std::vector<double> &A, const std::vector<double> &B, int n) {
  if (n == 0) {
    return std::vector<double>();
  }
  std::vector<double> C(n * n, 0.0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
  return C;
}