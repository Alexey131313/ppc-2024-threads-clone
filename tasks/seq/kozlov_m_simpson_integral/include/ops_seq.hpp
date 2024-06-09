// Copyright 2024 Kozlov Mikhail
#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using FUNC = double (*)(double, double);

double xy(double x, double y);
double sinxy(double x, double y);
double linear(double x, double y);
double expxy(double x, double y);
double expy_x(double x, double y);

class KozlovTaskSequential : public ppc::core::Task {
 public:
  explicit KozlovTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  FUNC f{};
  double x1{}, x2{}, y1{}, y2{};
  uint64_t n;
  uint64_t m;
  double res;
};