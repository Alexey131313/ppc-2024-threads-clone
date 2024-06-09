// Copyright 2023 Belan Vadim
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

class MatrixMulFox : public ppc::core::Task {
 public:
  explicit MatrixMulFox(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int n_{}, block_size_{};
  std::vector<double> matrix_a_, matrix_b_, matrix_c_;

  void fox_algorithm();
};
