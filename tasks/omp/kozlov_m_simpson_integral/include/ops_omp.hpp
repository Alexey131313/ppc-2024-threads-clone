// Copyright 2024 Kozlov Mikhail
#pragma once

#include <omp.h>

#include <cmath>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
namespace kozlov_omp {
using FUNC = double (*)(double, double);

double xy(double x, double y);
double sinxy(double x, double y);
double linear(double x, double y);
double expxy(double x, double y);
double expy_x(double x, double y);

class KozlovTasknOmpSequential : public ppc::core::Task {
 public:
  explicit KozlovTasknOmpSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  FUNC f{};
  double x1{}, x2{}, y1{}, y2{};
  int n;
  int m;
  double res;
};

class KozlovTasknOmpParallel : public ppc::core::Task {
 public:
  explicit KozlovTasknOmpParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  FUNC f{};
  double x1{}, x2{}, y1{}, y2{};
  int n;
  int m;
  double res;
};
}  // namespace kozlov_omp