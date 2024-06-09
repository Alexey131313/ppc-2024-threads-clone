// Copyright 2024 Videneva Ekaterina
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class Videneva_opm_task : public ppc::core::Task {
 public:
  explicit Videneva_opm_task(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double xLimL, xLimU, yLimL, yLimU, result, error, hX, hY;
  int64_t number;
};