// Copyright 2024 Eremin Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class RadixSortTaskOMP : public ppc::core::Task {
 public:
  explicit RadixSortTaskOMP(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> VectorForSort;
  int SizeVector = 0;
};
