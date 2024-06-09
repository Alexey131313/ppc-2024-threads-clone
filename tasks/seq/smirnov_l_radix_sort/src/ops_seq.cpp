// Copyright 2024 Smirnov Leonid
#include "seq/smirnov_l_radix_sort/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>

bool RadixSortSequential::validation() {
  internal_order_test();
  // Check count elements of output
  if (taskData->inputs_count.size() != 1 || taskData->outputs_count.size() != 1 || taskData->inputs.size() != 1 ||
      taskData->outputs.size() != 1) {
    return false;
  }
  if (taskData->inputs[0] == nullptr || taskData->outputs[0] == nullptr) {
    return false;
  }
  if (taskData->inputs_count[0] != taskData->outputs_count[0]) {
    return false;
  }
  return true;
}

bool RadixSortSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  try {
    auto size = taskData->inputs_count[0];
    auto* data = reinterpret_cast<int*>(taskData->inputs[0]);
    work_vector = std::vector(data, data + size);
    return true;
  } catch (...) {
    return false;
  }
}

bool RadixSortSequential::run() {
  internal_order_test();
  try {
    std::vector<int> temporaryArray = work_vector;

    int maxElem = *std::max_element(temporaryArray.begin(), temporaryArray.end());
    int numDigitsMaxElement = maxElem == 0 ? 1 : static_cast<int>(log10(abs(maxElem))) + 1;

    for (int digitIndex = 0; digitIndex < numDigitsMaxElement; digitIndex++) {
      int devider = static_cast<int>(pow(10, digitIndex));
      int minDigit = temporaryArray[0] % (devider * 10) / devider;
      int maxDigit = minDigit;

      for (size_t i = 0; i < temporaryArray.size(); i++) {
        int currentDigit = temporaryArray[i] % (devider * 10) / devider;
        minDigit = std::min(minDigit, currentDigit);
        maxDigit = std::max(maxDigit, currentDigit);
      }

      std::vector<int> frequency(maxDigit - minDigit + 1);
      for (size_t i = 0; i < temporaryArray.size(); i++) {
        int keyCurrentDigit = temporaryArray[i] % (devider * 10) / devider;
        frequency[keyCurrentDigit - minDigit]++;
      }

      int cumulativeSum = 0;
      for (int& c : frequency) {
        cumulativeSum += c;
        c = cumulativeSum;
      }

      std::vector<int> tempSorted(temporaryArray.size());
      for (int i = static_cast<int>(temporaryArray.size()) - 1; i >= 0; i--) {
        int key = temporaryArray[i] % (devider * 10) / devider;
        tempSorted[--frequency[key - minDigit]] = temporaryArray[i];
      }
      temporaryArray = tempSorted;
    }
    work_vector = temporaryArray;
    return true;
  } catch (...) {
    return false;
  }
}

bool RadixSortSequential::post_processing() {
  internal_order_test();
  auto* outputs = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < work_vector.size(); i++) {
    outputs[i] = work_vector[i];
  }
  return true;
}
