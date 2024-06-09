// Copyright 2024 Travin Maksim
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

using function = double (*)(double, double);

class SimpsonTask : public ppc::core::Task {
 public:
  explicit SimpsonTask(std::shared_ptr<ppc::core::TaskData> taskData_, function fun_)
      : Task(std::move(taskData_)), func(std::move(fun_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  function func;
  double res{};
  int a{}, b{}, c{}, d{}, n{};
};

double LinFunc(double x, double y);
double TrigFunc(double x, double y);
double MulFunc(double x, double y);
double ExpFunc(double x, double y);
double RandFunc(double x, double y);