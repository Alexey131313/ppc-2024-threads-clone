// Copyright 2024 Kiselev Igor
#ifndef SHELL_SIMPLE_HPP_INCLUDED
#define SHELL_SIMPLE_HPP_INCLUDED
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace Kiselev_seq {
class KiselevTaskSequential : public ppc::core::Task {
 public:
  explicit KiselevTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> arr;
  std::vector<int> input_;
};
}  // namespace Kiselev_seq
#endif  // SHELL_SIMPLE_HPP_INCLUDED
