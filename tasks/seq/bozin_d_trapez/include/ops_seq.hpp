// Copyright 2024 Bozin Dmitry
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class BozinTaskSequential : public ppc::core::Task {
 public:
  explicit BozinTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_,
                               std::function<double(double, double)> func_)
      : Task(std::move(taskData_)), func(std::move(func_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double res = 0.0;
  double ax{}, bx{}, ay{}, by{};
  int ny{}, nx{};
  std::function<double(double, double)> func;
};

double multiDimensionalIntegral(const std::function<double(double, double)>& func, double ax, double bx, double ay,
                                double by, int nx, int ny);