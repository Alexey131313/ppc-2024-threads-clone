// Copyright 2024 Andrey Derun
#include "tbb/derun_a_shell/include/shell_tbb.hpp"

#include <oneapi/tbb/parallel_for.h>

#include <thread>

using namespace std::chrono_literals;

bool ShellTBB::pre_processing() {
  internal_order_test();
  // Init value for input and output
  for (int i = 0; i < static_cast<int>(taskData->inputs_count[0]); i++) {
    input_.push_back(reinterpret_cast<int*>(taskData->inputs[0])[i]);
  }
  // res = 0;
  return true;
}

bool ShellTBB::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count.size() == 1 && taskData->inputs_count[0] > 0 &&
         taskData->outputs_count.size() == 1 &&  // NOLINT
         taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool ShellTBB::run() {
  internal_order_test();
  shell_sort_parallel(input_);
  // std::this_thread::sleep_for(20ms);
  return true;
}

bool ShellTBB::post_processing() {
  internal_order_test();
  // reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  for (int i = 0; i < static_cast<int>(taskData->inputs_count[0]); i++) {
    // input_.push_back(reinterpret_cast<int*>(taskData->inputs[0])[i]);
    reinterpret_cast<int*>(taskData->outputs[0])[i] = input_[i];
  }
  return true;
}

std::vector<int> ShellTBB::merge(const std::vector<std::vector<int>>& chunks) {
  std::vector<int> result;

  // Merge the sorted chunks
  for (const auto& chunk : chunks) {
    // Merge the current chunk with the result vector
    std::vector<int> merged;
    merged.reserve(result.size() + chunk.size());
    auto resultIter = result.begin();
    auto chunkIter = chunk.begin();

    while (resultIter != result.end() && chunkIter != chunk.end()) {
      if (*resultIter < *chunkIter) {
        merged.push_back(*resultIter);
        ++resultIter;
      } else {
        merged.push_back(*chunkIter);
        ++chunkIter;
      }
    }

    // Copy remaining elements from result vector
    while (resultIter != result.end()) {
      merged.push_back(*resultIter);
      ++resultIter;
    }

    // Copy remaining elements from current chunk
    while (chunkIter != chunk.end()) {
      merged.push_back(*chunkIter);
      ++chunkIter;
    }

    // Update result vector with merged vector
    result = std::move(merged);
  }

  return result;
}

void ShellTBB::shell_sort_parallel(std::vector<int>& input) {
  int numProcs = 4;
  // int numProcs = tbb::task_scheduler_init::default_num_threads();
  int size = static_cast<int>(input.size());
  int chunkSize = size / numProcs;
  int remainder = size % numProcs;

  std::vector<std::vector<int>> chunks(numProcs);

  tbb::parallel_for(0, numProcs, [&](int i) {
    int startIdx = i * chunkSize;
    int endIdx = startIdx + chunkSize;
    if (i == numProcs - 1) {
      endIdx += remainder;  // Add remaining elements to the last chunk
    }
    chunks[i].assign(input.begin() + startIdx, input.begin() + endIdx);
    shell_sort(chunks[i]);
  });

  // Barrier is not needed in TBB

  input = merge(chunks);
}

void ShellTBB::shell_sort(std::vector<int>& vec) {
  for (int interval = static_cast<int>(vec.size()) / 2; interval > 0; interval /= 2) {
    for (int i = interval; i < static_cast<int>(vec.size()); i++) {
      int tmp = vec[i];
      int j = i;
      for (; j >= interval && vec[j - interval] > tmp; j -= interval) {
        vec[j] = vec[j - interval];
      }
      vec[j] = tmp;
    }
  }
}

bool ShellTBB::checkSorted(std::vector<int> input) { return std::is_sorted(input.begin(), input.end()); }

std::vector<int> ShellTBB::generate_random_vector(int size, int min, int max) {
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()};  // Generates random integers
  std::uniform_int_distribution<int> dist{min, max};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

  std::vector<int> vec(size);
  generate(begin(vec), end(vec), gen);

  return vec;
}
