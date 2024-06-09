// Copyright 2024 Balyasov Ivan
#include "seq/balyasov_i_sort_simple_merge/include/ops_seq.hpp"

#include <cmath>
#include <thread>

using namespace std::chrono_literals;

bool RadixSortSimpleMergeTaskSequential::pre_processing() {
  internal_order_test();

  auto* int_data = reinterpret_cast<int*>(taskData->inputs[0]);
  SizeVector = taskData->inputs_count[0];
  VectorForSort.resize(SizeVector);
  std::copy(int_data, int_data + SizeVector, VectorForSort.begin());
  return true;
}

bool RadixSortSimpleMergeTaskSequential::validation() {
  internal_order_test();

  const bool inputs_valid = taskData->inputs[0] != nullptr && taskData->inputs_count[0] != 0;
  const bool outputs_valid = taskData->outputs[0] != nullptr;
  const bool counts_match = taskData->inputs_count[0] == taskData->outputs_count[0];

  return inputs_valid && outputs_valid && counts_match;
}

bool RadixSortSimpleMergeTaskSequential::run() {
  internal_order_test();

  std::vector<int> copiedVectorForSort = VectorForSort;
  int numDevider = 0;
  int maxNum = 0;
  int MinKey;
  int MaxKey;

  for (int num : copiedVectorForSort) {
    maxNum = std::max(maxNum, num);
  }

  numDevider = maxNum == 0 ? 1 : static_cast<int>(log10(abs(maxNum))) + 1;

  for (int dev = 0; dev < numDevider; dev++) {
    int devider = static_cast<int>(pow(10, dev));

    MinKey = copiedVectorForSort[0] % (devider * 10) / devider;
    MaxKey = MinKey;

    for (size_t i = 0; i < copiedVectorForSort.size(); i++) {
      int digit = copiedVectorForSort[i] % (devider * 10) / devider;
      MinKey = std::min(MinKey, digit);
      MaxKey = std::max(MaxKey, digit);
    }

    std::vector<int> count(MaxKey - MinKey + 1);
    for (size_t i = 0; i < copiedVectorForSort.size(); i++) {
      int key = copiedVectorForSort[i] % (devider * 10) / devider;
      count[key - MinKey]++;
    }

    int running_sum = 0;
    for (int& c : count) {
      running_sum += c;
      c = running_sum;
    }

    std::vector<int> temp(copiedVectorForSort.size());
    for (int i = static_cast<int>(copiedVectorForSort.size()) - 1; i >= 0; i--) {
      int key = copiedVectorForSort[i] % (devider * 10) / devider;
      temp[--count[key - MinKey]] = copiedVectorForSort[i];
    }

    copiedVectorForSort = temp;
  }

  VectorForSort = copiedVectorForSort;

  return true;
}

bool RadixSortSimpleMergeTaskSequential::post_processing() {
  internal_order_test();

  auto* outputs = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < VectorForSort.size(); i++) {
    outputs[i] = VectorForSort[i];
  }

  return true;
}
