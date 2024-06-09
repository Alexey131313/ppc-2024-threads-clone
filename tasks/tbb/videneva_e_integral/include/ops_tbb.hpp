// Copyright 2024 Videneva Ekaterina
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace Videneva_e_tbb_integral {

class VidenevaEIntegralTBB : public ppc::core::Task {
 public:
  explicit VidenevaEIntegralTBB(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double xLimL, xLimU, yLimL, yLimU, result, error;
  int64_t number;
};

}  // namespace Videneva_e_tbb_integral