// Copyright 2024 Alexseev Danila
#include "omp/alexseev_d_hoare_sort_w_batcher_merge/include/hoare_sort_w_batcher_merge_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <random>
#include <thread>

using namespace std::chrono_literals;

// Sequential implementation of sorting
bool alexseev_omp::HoareSortWBatcherMergeSequential::pre_processing() {
  try {
    internal_order_test();
    array.clear();
    for (size_t i = 0; i < taskData->inputs_count[0]; ++i) {
      int *currentElementPtr = reinterpret_cast<int *>(taskData->inputs[0] + i * sizeof(int));
      array.push_back(*currentElementPtr);
    }
  } catch (...) {
    return false;
  }
  return true;
}

bool alexseev_omp::HoareSortWBatcherMergeSequential::validation() {
  try {
    internal_order_test();
  } catch (...) {
    return false;
  }
  return taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool alexseev_omp::HoareSortWBatcherMergeSequential::run() {
  try {
    internal_order_test();
    HoareSortWBatcherMergeSeq(array, 0, array.size() - 1);
  } catch (...) {
    return false;
  }
  return true;
}

bool alexseev_omp::HoareSortWBatcherMergeSequential::post_processing() {
  try {
    internal_order_test();
    for (size_t i = 0; i < array.size(); ++i) {
      int *currentElementPtr = reinterpret_cast<int *>(taskData->outputs[0] + i * sizeof(int));
      *currentElementPtr = array[i];
    }
  } catch (...) {
    return false;
  }
  return true;
}

void alexseev_omp::HoareSortWBatcherMergeSequential::HoareSortWBatcherMergeSeq(std::vector<int> &arr, size_t l,
                                                                               size_t r) {
  if (arr.size() <= 1) return;
  int n = r - l + 1;
  for (int p = 1; p < n; p += p)
    for (int k = p; k > 0; k /= 2)
      for (int j = k % p; j + k < n; j += (k + k))
        for (int i = 0; i < n - j - k; ++i)
          if ((j + i) / (p + p) == (j + i + k) / (p + p)) CompExch(arr[l + j + i], arr[l + j + i + k]);
}

// OMP implementation of sorting
bool alexseev_omp::HoareSortWBatcherMergeOMP::pre_processing() {
  try {
    internal_order_test();
    array.clear();
    for (size_t i = 0; i < taskData->inputs_count[0]; ++i) {
      int *currentElementPtr = reinterpret_cast<int *>(taskData->inputs[0] + i * sizeof(int));
      array.push_back(*currentElementPtr);
    }
  } catch (...) {
    return false;
  }
  return true;
}

bool alexseev_omp::HoareSortWBatcherMergeOMP::validation() {
  try {
    internal_order_test();
  } catch (...) {
    return false;
  }
  return taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool alexseev_omp::HoareSortWBatcherMergeOMP::run() {
  try {
    internal_order_test();
    HoareSortWBatcherMergeParallel(array, 0, array.size() - 1);
  } catch (...) {
    return false;
  }
  return true;
}

bool alexseev_omp::HoareSortWBatcherMergeOMP::post_processing() {
  try {
    internal_order_test();
    if (array.size() != taskData->outputs_count[0]) {
      throw;
    }
    for (size_t i = 0; i < array.size(); ++i) {
      int *currentElementPtr = reinterpret_cast<int *>(taskData->outputs[0] + i * sizeof(int));
      *currentElementPtr = array[i];
    }
  } catch (...) {
    return false;
  }
  return true;
}

void alexseev_omp::HoareSortWBatcherMergeOMP::HoareSortWBatcherMergeParallel(std::vector<int> &arr, size_t l,
                                                                             size_t r) {
  if (arr.size() <= 1) return;
  int n = r - l + 1;

  for (int p = 1; p < n; p += p)
    for (int k = p; k > 0; k /= 2)
      for (int j = k % p; j + k < n; j += (k + k))
#pragma omp parallel for
        for (int i = 0; i < n - j - k; ++i)
          if ((j + i) / (p + p) == (j + i + k) / (p + p)) {
            CompExch(arr[l + j + i], arr[l + j + i + k]);
          }
}

// Additional functions
void alexseev_omp::CompExch(int &a, int &b) {
  if (a > b) std::swap(a, b);
}
