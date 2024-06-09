// Copyright 2024 Nesterov Alexander
#include "seq/benduyzhko_t_shell_batcher/include/ops_seq.hpp"

#include <iostream>
#include <random>
#include <vector>

bool BenduyzhkoSequential::pre_processing() {
  internal_order_test();
  n = taskData->inputs_count[0];
  in_out = reinterpret_cast<int*>(taskData->outputs[0]);
  return true;
}

bool BenduyzhkoSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->outputs_count[0] && taskData->inputs_count[0] != 0u;
}

void shell(int* arr, int n);

bool BenduyzhkoSequential::run() {
  internal_order_test();
  int p = n >> 1;  // n = 2 * p;
  shell(in_out, p);
  shell(in_out + p, n - p);
  for (int k = p; k >= 1; k >>= 1) {
    for (int j = k % p; j < n - k + 1; j += 2 * k) {
      for (int i = 0; i < std::min(k - 1, n - j - k - 1) + 1; ++i) {
        if ((i + j) % n + k < n) {  // if ((i + j) / n == (i + j + k) / n) {
          if (in_out[i + j] > in_out[i + j + k]) {
            std::swap(in_out[i + j], in_out[i + j + k]);
          }
        }
      }
    }
  }
  return true;
}

bool BenduyzhkoSequential::post_processing() {
  internal_order_test();
  return true;
}

void shell(int* arr, int n) {
  int tmp;
  int j;
  for (int gap = n >> 1; gap > 0; gap >>= 1) {
    for (int i = gap; i < n; ++i) {
      tmp = arr[i];
      for (j = i; j >= gap && arr[j - gap] > tmp; j -= gap) {
        arr[j] = arr[j - gap];
      }
      arr[j] = tmp;
    }
  }
}

void BenduyzhkoSequential::get_random_numbers(int* arr, int n, int a, int b, int seed) {
  std::uniform_int_distribution<int> dist(a, b);
  std::random_device dev;
  std::mt19937 gen(seed != 0 ? seed : dev());
  for (int i = 0; i < n; i++) {
    arr[i] = dist(gen);
  }
}
