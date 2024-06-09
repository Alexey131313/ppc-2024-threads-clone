// Copyright 2024 Morgachev Stepan
#include "seq/morgachev_s_radix_sort_double/include/ops_seq.hpp"

#include <algorithm>
#include <thread>

using namespace std::chrono_literals;

bool RadixSortDoubleSequential::pre_processing() {
  internal_order_test();
  try {
    dataSize = taskData->inputs_count[0];
    auto* ptrToData = reinterpret_cast<double*>(taskData->inputs[0]);
    data.resize(dataSize);
    std::copy(ptrToData, ptrToData + dataSize, data.begin());
  } catch (...) {
    return false;
  }
  return true;
}

bool RadixSortDoubleSequential::validation() {
  internal_order_test();
  return taskData->inputs[0] != nullptr && taskData->outputs[0] != nullptr && taskData->inputs_count[0] != 0 &&
         taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool RadixSortDoubleSequential::run() {
  internal_order_test();
  try {
    std::vector<double> tempLeft = data;
    std::vector<double> tempRight(dataSize);

    for (int i = 0; i < 4; i++) {
      countSort(tempLeft.data(), tempRight.data(), dataSize, 2 * i);
      countSort(tempRight.data(), tempLeft.data(), dataSize, 2 * i + 1);
    }

    data = tempLeft;
  } catch (...) {
    return false;
  }
  return true;
}

bool RadixSortDoubleSequential::post_processing() {
  internal_order_test();
  try {
    for (size_t i = 0; i < data.size(); i++) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = data[i];
    }
  } catch (...) {
    return false;
  }
  return true;
}

void RadixSortDoubleSequential::countSort(double* in, double* out, int length, int exp) {
  auto* buffer = reinterpret_cast<unsigned char*>(in);
  int counter[256] = {0};
  int minValue = 0;
  int minPos = 0;

  for (int i = 0; i < length; i++) {
    counter[buffer[8 * i + exp]]++;
  }

  for (int j = 0; j < 256; j++) {
    if (counter[j] != 0) {
      minValue = counter[j];
      counter[j] = 0;
      minPos = ++j;
      break;
    }
  }

  for (; minPos < 256; minPos++) {
    int temp = counter[minPos];
    counter[minPos] = minValue;
    minValue += temp;
  }

  for (int i = 0; i < length; i++) {
    out[counter[buffer[8 * i + exp]]] = in[i];
    counter[buffer[8 * i + exp]]++;
  }
}
