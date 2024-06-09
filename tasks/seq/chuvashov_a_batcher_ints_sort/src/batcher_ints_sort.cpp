// Copyright 2024 Chuvashov Andrey

#include "seq/chuvashov_a_batcher_ints_sort/include/batcher_ints_sort.hpp"
using namespace std::chrono_literals;

std::vector<int> Chuvashov_BatcherEven(std::vector<int> arr1, std::vector<int> arr2) {
  std::vector<int> result(arr1.size() / 2 + arr2.size() / 2 + arr1.size() % 2 + arr2.size() % 2);
  size_t i = 0;
  size_t j = 0;
  size_t k = 0;

  while ((j < arr1.size()) && (k < arr2.size())) {
    if (arr1[j] <= arr2[k]) {
      result[i] = arr1[j];
      j += 2;
    } else {
      result[i] = arr2[k];
      k += 2;
    }
    i++;
  }

  if (j >= arr1.size()) {
    for (size_t l = k; l < arr2.size(); l += 2) {
      result[i] = arr2[l];
      i++;
    }
  } else {
    for (size_t l = j; l < arr1.size(); l += 2) {
      result[i] = arr1[l];
      i++;
    }
  }

  return result;
}

std::vector<int> Chuvashov_BatcherOdd(std::vector<int> arr1, std::vector<int> arr2) {
  std::vector<int> result(arr1.size() / 2 + arr2.size() / 2);
  size_t i = 0;
  size_t j = 1;
  size_t k = 1;

  while ((j < arr1.size()) && (k < arr2.size())) {
    if (arr1[j] <= arr2[k]) {
      result[i] = arr1[j];
      j += 2;
    } else {
      result[i] = arr2[k];
      k += 2;
    }
    i++;
  }

  if (j >= arr1.size()) {
    for (size_t l = k; l < arr2.size(); l += 2) {
      result[i] = arr2[l];
      i++;
    }
  } else {
    for (size_t l = j; l < arr1.size(); l += 2) {
      result[i] = arr1[l];
      i++;
    }
  }

  return result;
}

std::vector<int> Chuvashov_merge(std::vector<int> arr1, std::vector<int> arr2) {
  std::vector<int> result(arr1.size() + arr2.size());
  size_t i = 0;
  size_t j = 0;
  size_t k = 0;

  while ((j < arr1.size()) && (k < arr2.size())) {
    result[i] = arr1[j];
    result[i + 1] = arr2[k];
    i += 2;
    j++;
    k++;
  }

  if ((k >= arr2.size()) && (j < arr1.size())) {
    for (size_t l = i; l < result.size(); l++) {
      result[l] = arr1[j];
      j++;
    }
  }

  for (size_t l = 0; l < result.size() - 1; l++) {
    if (result[l] > result[l + 1]) {
      std::swap(result[l], result[l + 1]);
    }
  }

  return result;
}

std::vector<int> Chuvashov_BatcherSort(const std::vector<int> &arr1, const std::vector<int> &arr2) {
  std::vector<int> even = Chuvashov_BatcherEven(arr1, arr2);
  std::vector<int> odd = Chuvashov_BatcherOdd(arr1, arr2);
  std::vector<int> result = Chuvashov_merge(even, odd);
  return result;
}

bool Chuvashov_SequentialBatcherSort::pre_processing() {
  internal_order_test();
  input = std::vector<int>(taskData->inputs_count[0]);
  auto *tmp_ptr_A = reinterpret_cast<int *>(taskData->inputs[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    input[i] = tmp_ptr_A[i];
  }
  arr1.resize(input.size() / 2);
  arr2.resize(input.size() / 2);

  for (size_t i = 0; i < (input.size() / 2); i++) {
    arr1[i] = input[i];
    arr2[i] = input[(input.size() / 2) + i];
  }

  std::sort(arr1.begin(), arr1.end());
  std::sort(arr2.begin(), arr2.end());
  return true;
}

bool Chuvashov_SequentialBatcherSort::validation() {
  internal_order_test();
  return std::is_sorted(arr1.begin(), arr1.end()) && std::is_sorted(arr2.begin(), arr2.end());
}

bool Chuvashov_SequentialBatcherSort::run() {
  internal_order_test();
  output = Chuvashov_BatcherSort(arr1, arr2);
  return true;
}

bool Chuvashov_SequentialBatcherSort::post_processing() {
  internal_order_test();
  std::copy(output.begin(), output.end(), reinterpret_cast<int *>(taskData->outputs[0]));
  return true;
}
