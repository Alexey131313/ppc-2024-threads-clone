// Copyright 2024 Ivanov Nikita
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

using func = double (*)(double, double);

class TestTBBTaskSequentialIvanovSimpson : public ppc::core::Task {
 public:
  explicit TestTBBTaskSequentialIvanovSimpson(std::shared_ptr<ppc::core::TaskData> taskData_, func fun_)
      : Task(std::move(taskData_)), fun(std::move(fun_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int a{}, b{}, c{}, d{}, n{};
  double res{};
  func fun;
};

class TestTBBTaskParallelIvanovSimpson : public ppc::core::Task {
 public:
  explicit TestTBBTaskParallelIvanovSimpson(std::shared_ptr<ppc::core::TaskData> taskData_, func fun_)
      : Task(std::move(taskData_)), fun(std::move(fun_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int a{}, b{}, c{}, d{}, n{};
  double res{};
  func fun;
};

double linear_fun(double x, double y);
double sin_cos(double x, double y);
double x_mul_y(double x, double y);