// Copyright 2024 Lysanova Julia
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

class LysanovaTaskSequential : public ppc::core::Task {
 public:
  explicit LysanovaTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double *A{}, *B{}, *C{};
  size_t n{};
};

void RandomFillMatrix(double* matrix, int sz);
void EMatrix(double* matrix, int n, double k = 1.0);
void EsdMatrix(double* matrix, int n, double k = 1.0);
