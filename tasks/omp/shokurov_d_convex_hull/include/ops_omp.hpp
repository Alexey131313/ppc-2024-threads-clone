// Copyright 2024 Shokurov Daniil
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
using namespace std;

class ConvexHullSequential : public ppc::core::Task {
 public:
  explicit ConvexHullSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static double normal(const pair<double, double>& v);
  static double scalar_product(const pair<double, double>& v1, const pair<double, double>& v2);

  static double cos(const pair<double, double>& v1, const pair<double, double>& v2);

  static bool my_less(const pair<double, double>& v1, const pair<double, double>& v2, const pair<double, double>& v);

  static size_t index_lowest_right_point(const vector<pair<double, double>>& v);

  static pair<double, double> sub(const pair<double, double>& v1, const pair<double, double>& v2);

  static bool comp(const pair<double, double>& a, const pair<double, double>& b);

  static size_t solve(vector<pair<double, double>>& p);

 private:
  vector<pair<double, double>> points;
  size_t si = 0;
};
