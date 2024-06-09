// Copyright 2024 Benduyzhko Tatiana
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

class BenduyzhkoSequential : public ppc::core::Task {
 public:
  explicit BenduyzhkoSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static void get_random_numbers(int*, int, int = -50, int = 50, int = 0);

 private:
  int* in_out{};
  int n{};
};
