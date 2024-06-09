// Copyright 2024 Shemiakina Alesia
#pragma once

#include <algorithm>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

class ShellTaskSequential : public ppc::core::Task {
 public:
  explicit ShellTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static bool CheckSort(std::vector<int> input);

  static std::vector<int> give_random_vector(int size, int min, int max);

 private:
  static std::vector<int> ShellSort(const std::vector<int>& input);
  std::vector<int> input_;
};
