// Copyright 2024 Shokurov Daniil
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shokurov_d_convex_hull_seq {
class ConvexHullShokurov : public ppc::core::Task {
 public:
  explicit ConvexHullShokurov(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static double normal(const std::pair<double, double>& v);
  static double scalar_product(const std::pair<double, double>& v1, const std::pair<double, double>& v2);

  static double cos(const std::pair<double, double>& v1, const std::pair<double, double>& v2);

  static bool my_less(const std::pair<double, double>& v1, const std::pair<double, double>& v2,
                      const std::pair<double, double>& v);

  static size_t index_lowest_right_point(const std::vector<std::pair<double, double>>& v);

  static std::pair<double, double> sub(const std::pair<double, double>& v1, const std::pair<double, double>& v2);

  static bool comp(const std::pair<double, double>& a, const std::pair<double, double>& b);

  static size_t solve(std::vector<std::pair<double, double>>& p);

 private:
  std::vector<std::pair<double, double>> points;
  size_t si = 0;
};
}  // namespace shokurov_d_convex_hull_seq