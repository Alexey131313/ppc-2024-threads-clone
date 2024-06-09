// Copyright 2024 Kashirin Alexander

#include "seq/kashirin_a_int_radix_sort_batcher/include/ops_seq.hpp"
using namespace std::chrono_literals;

int remainder(int num, int k) {
  while (k > 1) {
    num = num / 10;
    k--;
  }
  return num % 10;
}

void merge(std::vector<int>& a, size_t left, size_t mid, size_t right) {
  std::vector<int> b(a.size(), 0);
  size_t i = left;
  size_t j = mid + 1;
  size_t k = left;
  while (k <= right) {
    if (((j > right) || (a[i] < a[j])) && (i <= mid)) {
      b[k++] = a[i++];
    } else if (((i > mid) || (a[i] > a[j])) && (j <= right)) {
      b[k++] = a[j++];
    } else {
      b[k] = b[k + 1] = a[i++];
      k += 2;
      j++;
    }
  }
  for (size_t q = left; q <= right; q++) {
    a[q] = b[q];
  }
}

void sort(const std::vector<int>& src, std::vector<int>& result, size_t left, size_t right) {
  std::vector<int> b(src);
  std::vector<std::vector<int>> tmp(10, std::vector<int>(100, 0));
  std::vector<int> amount(10, 0);
  int k = 1;
  while (k <= 3) {
    for (size_t i = left; i <= right; i++) {
      int rem = remainder(b[i], k);
      tmp[rem][amount[rem]++] = b[i];
    }
    for (int i = 0, ind = left; i < 10; i++) {
      for (int j = 0; j < amount[i]; j++) {
        b[ind] = tmp[i][j];
        ind++;
      }
      amount[i] = 0;
    }
    k++;
  }
  for (size_t i = left; i <= right; i++) {
    result[i] = b[i];
  }
}

void recursiveSort(std::vector<int>& src, std::vector<int>& result, size_t left, size_t right) {
  if (right - left <= 8) {
    sort(src, result, left, right);
  } else {
    auto mid = (size_t)((right + left) / 2);
    recursiveSort(src, result, left, mid);
    recursiveSort(src, result, mid + 1, right);
    merge(result, left, mid, right);
  }
}

bool SeqIntRadixSortWithBatcherMerge::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input[i] = tmp[i];
  }
  result = std::vector<int>(taskData->outputs_count[0]);
  return true;
}

bool SeqIntRadixSortWithBatcherMerge::validation() {
  internal_order_test();

  // Check count elements of output
  return taskData->inputs_count[0] == taskData->outputs_count[0];
  // return taskData->inputs_count.size() == taskData->outputs_count.size();
  // return true;
}

bool SeqIntRadixSortWithBatcherMerge::run() {
  internal_order_test();
  try {
    recursiveSort(input, result, 0, input.size() - 1);
  } catch (...) {
    return false;
  }
  // std::this_thread::sleep_for(20ms);
  return true;
}

bool SeqIntRadixSortWithBatcherMerge::post_processing() {
  internal_order_test();
  std::copy(result.begin(), result.end(), reinterpret_cast<int*>(taskData->outputs[0]));

  return true;
  // return std::is_sorted(result.begin(), result.end());
}
