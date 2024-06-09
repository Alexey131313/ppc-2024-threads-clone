// Copyright 2024 Eremin Alexander
#include "seq/eremin_a_int_radixsort/include/ops_seq.hpp"

#include <cmath>
#include <thread>

using namespace std::chrono_literals;

bool RadixSortTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  try {
    int* int_data = reinterpret_cast<int*>(taskData->inputs[0]);
    SizeVector = taskData->inputs_count[0];
    VectorForSort.resize(SizeVector);
    std::copy(int_data, int_data + SizeVector, VectorForSort.begin());

  } catch (...) {
    return false;
  }
  return true;
}

bool RadixSortTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == taskData->outputs_count[0] && taskData->outputs[0] != nullptr &&
         taskData->inputs[0] != nullptr && taskData->inputs_count[0] != 0;
}

bool RadixSortTaskSequential::run() {
  internal_order_test();
  try {
    std::vector<int> VectorForSort_;
    VectorForSort_ = VectorForSort;
    int numDevider = 0;
    int maxNum = 0;
    int MinKey;
    int MaxKey;

    for (size_t i = 0; i < VectorForSort_.size(); i++) {
      if (maxNum <= VectorForSort_[i]) {
        maxNum = VectorForSort_[i];
      }
    }

    if (maxNum == 0) {
      numDevider = 1;
    } else
      numDevider = log10(abs(maxNum)) + 1;

    for (int dev = 0; dev < numDevider; dev++) {
      int devider = pow(10, dev);

      MinKey = VectorForSort_[0] % (devider * 10) / devider;
      MaxKey = MinKey;
      for (size_t i = 0; i < VectorForSort_.size(); i++) {
        int digit = VectorForSort_[i] % (devider * 10) / devider;
        if (digit < MinKey) {
          MinKey = digit;
        }
        if (digit > MaxKey) {
          MaxKey = digit;
        }
      }

      std::vector<int> count(MaxKey - MinKey + 1);
      for (size_t i = 0; i < VectorForSort_.size(); i++) {
        count[(VectorForSort_[i] % (devider * 10) / devider) - MinKey]++;
      }

      int size = VectorForSort_.size();
      for (int i = count.size() - 1; i >= 0; i--) {
        size -= count[i];
        count[i] = size;
      }
      std::vector<int> temp(VectorForSort_.size());
      for (size_t i = 0; i < VectorForSort_.size(); i++) {
        temp[count[(VectorForSort_[i] % (devider * 10) / devider) - MinKey]] = VectorForSort_[i];
        count[(VectorForSort_[i] % (devider * 10) / devider) - MinKey]++;
      }
      for (size_t i = 0; i < VectorForSort_.size(); i++) {
        VectorForSort_[i] = temp[i];
      }
    }
    VectorForSort = VectorForSort_;
  } catch (...) {
    return false;
  }
  return true;
}

bool RadixSortTaskSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < VectorForSort.size(); i++) reinterpret_cast<int*>(taskData->outputs[0])[i] = VectorForSort[i];
  return true;
}
