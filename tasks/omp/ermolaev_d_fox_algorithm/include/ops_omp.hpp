// Copyright 2024 Ermolaev Danila
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class FoxAlgorithmOMP : public ppc::core::Task {
 public:
  explicit FoxAlgorithmOMP(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double* matrix_A;
  double* matrix_B;
  double* matrix_C;
  size_t data_size;
};