// Copyright 2024 Shmelev Ivan
#pragma once
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

class ShmelevTaskSequential : public ppc::core::Task {
 public:
  explicit ShmelevTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static std::vector<int> generate_random_vector(int size, int min, int max);
  static bool sorted(std::vector<int> input);

 private:
  std::vector<int> input_;
  std::vector<int> res;
  void batcherMerge(int l, int r);
  void sortingShell();
  void merge(int l, int m, int r);
};