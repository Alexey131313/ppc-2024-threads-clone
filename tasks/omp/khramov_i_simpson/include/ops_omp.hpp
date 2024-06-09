// Copyright 2024 Khramov Ivan
#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace khramov_omp {

using function = double (*)(double, double);

class TestSimpsonOmpSequential : public ppc::core::Task {
 public:
  explicit TestSimpsonOmpSequential(std::shared_ptr<ppc::core::TaskData> taskData_, function func_)
      : Task(std::move(taskData_)), func(std::move(func_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int a1{}, a2{}, b1{}, b2{}, num_steps{};
  double result{};
  function func;
};

class TestSimpsonOmpParallel : public ppc::core::Task {
 public:
  explicit TestSimpsonOmpParallel(std::shared_ptr<ppc::core::TaskData> taskData_, function func_)
      : Task(std::move(taskData_)), func(std::move(func_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int a1{}, a2{}, b1{}, b2{}, num_steps{};
  double result{};
  function func;
};
double simpson_formula(function func, double Xj0, double Xj1, double Xi);
double simpson_seq(function func, int a1, int a2, int b1, int b2, int numSteps);
double simpson_parallel(function func, int a1, int a2, int b1, int b2, int numSteps);
double linear_f(double x0, double x1);
double square_f(double x0, double x1);
double sin_f(double x0, double x1);
double mult_f(double x0, double x1);
double exp_f(double x0, double x1);

}  // namespace khramov_omp
