// Copyright 2024 Kurdina Julia
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class HorizontalSplitSequential : public ppc::core::Task {
 public:
  explicit HorizontalSplitSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  uint8_t NewColor(int i, int j);

 private:
  std::vector<uint8_t> input_ = {};
  std::vector<uint8_t> image = {};
  std::vector<uint8_t> gauss_kernel = {};
  int n{}, m{};
  int ker_size = 3;
};
