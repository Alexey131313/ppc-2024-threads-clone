// Copyright 2024 Morgachev Stepan
#pragma once

#include <vector>

#include "core/task/include/task.hpp"

class RadixSortDoubleSequential : public ppc::core::Task {
 public:
  explicit RadixSortDoubleSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> data;
  size_t dataSize;
  static void countSort(double* in, double* out, int length, int exp);
};