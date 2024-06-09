// Copyright 2024 Andrey Derun
#include "seq/derun_a_shell/include/shell_seq.hpp"

#include <thread>  // NOLINT

using namespace std::chrono_literals;  // NOLINT

bool ShellSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  for (int i = 0; i < static_cast<int>(taskData->inputs_count[0]); i++) {
    input_.push_back(reinterpret_cast<int*>(taskData->inputs[0])[i]);
  }
  // res = 0;
  return true;
}

bool ShellSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count.size() == 1 && taskData->inputs_count[0] > 0 &&
         taskData->outputs_count.size() == 1 &&  // NOLINT
         taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool ShellSequential::run() {
  internal_order_test();
  input_ = shell_sort(input_);
  // std::this_thread::sleep_for(20ms);
  return true;
}

bool ShellSequential::post_processing() {
  internal_order_test();
  // reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  for (int i = 0; i < static_cast<int>(taskData->inputs_count[0]); i++) {
    // input_.push_back(reinterpret_cast<int*>(taskData->inputs[0])[i]);
    reinterpret_cast<int*>(taskData->outputs[0])[i] = input_[i];
  }
  return true;
}

std::vector<int> ShellSequential::shell_sort(const std::vector<int>& input) {
  std::vector<int> vec(input);

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

  return vec;
}

bool ShellSequential::checkSorted(std::vector<int> input) { return std::is_sorted(input.begin(), input.end()); }

std::vector<int> ShellSequential::generate_random_vector(int size, int min, int max) {
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
