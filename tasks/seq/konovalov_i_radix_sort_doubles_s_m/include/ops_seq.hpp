// Copyright 2024 Konovalov Igor
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class RadixSortSequentialTask : public ppc::core::Task {
 public:
  explicit RadixSortSequentialTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  size_t input_size;
  double* input_;
  std::vector<double> ordered[sizeof(double) * 256];
};
