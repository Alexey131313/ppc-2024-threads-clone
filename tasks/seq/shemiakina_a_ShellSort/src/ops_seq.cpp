// Copyright 2024 Shemiakina Alesya
#include "seq/shemiakina_a_ShellSort/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool ShellTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  for (int i = 0; i < static_cast<int>(taskData->inputs_count[0]); i++) {
    input_.push_back(reinterpret_cast<int*>(taskData->inputs[0])[i]);
  }
  return true;
}

bool ShellTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count.size() == 1 && taskData->inputs_count[0] > 0 && taskData->outputs_count.size() == 1 &&
         taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool ShellTaskSequential::run() {
  internal_order_test();
  input_ = ShellSort(input_);
  return true;
}

bool ShellTaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < static_cast<int>(taskData->inputs_count[0]); i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = input_[i];
  }
  return true;
}

std::vector<int> ShellTaskSequential::ShellSort(const std::vector<int>& input) {
  std::vector<int> vec(input);

  for (int interval = static_cast<int>(vec.size()) / 2; interval > 0; interval /= 2) {
    for (int i = interval; i < static_cast<int>(vec.size()); i++) {
      int tmp = vec[i];
      int j = i;
      for (; j >= interval && vec[j - interval] > tmp; j = j - interval) {
        vec[j] = vec[j - interval];
      }
      vec[j] = tmp;
    }
  }

  return vec;
}

bool ShellTaskSequential::CheckSort(std::vector<int> input) { return std::is_sorted(input.begin(), input.end()); }

std::vector<int> ShellTaskSequential::give_random_vector(int size, int min, int max) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_int_distribution<int> dist{min, max};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

  std::vector<int> vec(size);
  generate(begin(vec), end(vec), gen);

  return vec;
}
