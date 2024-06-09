// Copyright 2024 Konovalov Igor
#include "seq/konovalov_i_radix_sort_doubles_s_m/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool RadixSortSequentialTask::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = reinterpret_cast<double*>(taskData->inputs[0]);
  input_size = taskData->inputs_count[0];
  return true;
}

bool RadixSortSequentialTask::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count.size() == taskData->outputs_count.size();
}

bool RadixSortSequentialTask::run() {
  internal_order_test();
  try {
    char* buf;
    for (size_t i = 0; i < sizeof(double); i++) {
      for (size_t j = 0; j < input_size; j++) {
        buf = reinterpret_cast<char*>(&input_[j]);
        ordered[(uint8_t)buf[i]].push_back(input_[j]);
      }

      size_t cur = 0;

      for (size_t j = 0; j < sizeof(double) * 256; j++) {
        for (size_t k = 0; k < ordered[j].size(); k++) {
          input_[cur] = ordered[j][k];
          cur++;
        }
        ordered[j].clear();
      }
    }
  } catch (std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return false;
  }
  return true;
}

bool RadixSortSequentialTask::post_processing() {
  internal_order_test();
  *reinterpret_cast<double**>(taskData->outputs[0]) = input_;
  return true;
}
