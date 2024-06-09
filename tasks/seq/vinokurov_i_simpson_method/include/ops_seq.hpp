// Copyright 2024 Vinokurov Ivan
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

using func = double (*)(double, double);

double fn_linear(double x, double y);
double fn_exp(double x, double y);
double fn_trig(double x, double y);
double fn_mul(double x, double y);
double fn_other(double x, double y);

class SimpsonMethodSequential : public ppc::core::Task {
 public:
  explicit SimpsonMethodSequential(std::shared_ptr<ppc::core::TaskData> taskData_, func fn_)
      : Task(std::move(taskData_)), fn(std::move(fn_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int a{}, b{}, c{}, d{}, n{};
  double result{};
  func fn;
};
