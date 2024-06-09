// Copyright 2024 Shmelev Ivan

#include "seq/shmelev_i_shell_sorting_with_Batcher/include/ops_seq.hpp"

#include <algorithm>

bool ShmelevTaskSequential::pre_processing() {
  internal_order_test();
  input_ = *reinterpret_cast<std::vector<int>*>(taskData->inputs[0]);
  res = *reinterpret_cast<std::vector<int>*>(taskData->outputs[0]);
  return true;
}

bool ShmelevTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool ShmelevTaskSequential::run() {
  internal_order_test();
  batcherMerge(0, input_.size() - 1);
  return true;
}

bool ShmelevTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<std::vector<int>*>(taskData->outputs[0])->operator=(input_);

  return true;
}

std::vector<int> ShmelevTaskSequential::generate_random_vector(int size, int min, int max) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_int_distribution<int> dist{min, max};

  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

  std::vector<int> vector(size);
  generate(begin(vector), end(vector), gen);

  return vector;
}

void ShmelevTaskSequential::sortingShell() {
  int n = input_.size();
  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; i++) {
      int temp = input_[i];
      int j;
      for (j = i; j >= gap && input_[j - gap] > temp; j -= gap) {
        input_[j] = input_[j - gap];
      }
      input_[j] = temp;
    }
  }
}

void ShmelevTaskSequential::batcherMerge(int l, int r) {
  sortingShell();
  if (r > l) {
    int m = l + (r - l) / 2;
    for (int i = l; i <= m; i += 2) {
      if (i + 1 <= m) {
        if (input_[i] > input_[i + 1]) {
          std::swap(input_[i], input_[i + 1]);
        }
      }
    }
    batcherMerge(l, m);
    batcherMerge(m + 1, r);
    merge(l, m, r);
  }
}
bool ShmelevTaskSequential::sorted(std::vector<int> input) { return std::is_sorted(input.begin(), input.end()); }

void ShmelevTaskSequential::merge(int l, int m, int r) {
  int n1 = m - l + 1;
  int n2 = r - m;

  std::vector<int> L(n1);
  std::vector<int> R(n2);

  for (int i = 0; i < n1; i++) {
    L[i] = input_[l + i];
  }
  for (int j = 0; j < n2; j++) {
    R[j] = input_[m + 1 + j];
  }

  int i = 0;
  int j = 0;
  int k = l;
  while (i < n1 && j < n2) {
    if (L[i] <= R[j]) {
      input_[k] = L[i];
      i++;
    } else {
      input_[k] = R[j];
      j++;
    }
    k++;
  }

  while (i < n1) {
    input_[k] = L[i];
    i++;
    k++;
  }
  while (j < n2) {
    input_[k] = R[j];
    j++;
    k++;
  }
}