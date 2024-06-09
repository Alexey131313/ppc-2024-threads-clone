// Copyright 2024 Andrey Derun
#pragma once

#include <omp.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

class ShellTBB : public ppc::core::Task {
 public:
  explicit ShellTBB(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}  // NOLINT
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static bool checkSorted(std::vector<int> input);

  static std::vector<int> generate_random_vector(int size, int min, int max);

 private:
  static std::vector<int> merge(const std::vector<std::vector<int>>& chunks);
  static void shell_sort_parallel(std::vector<int>& input);
  static void shell_sort(std::vector<int>& vec);
  std::vector<int> input_;
};
