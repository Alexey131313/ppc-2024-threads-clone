// Copyright 2024 Durandin Vladimir
#pragma once

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace Const {
constexpr double MY_PI = 3.14159265358980;
}  //  namespace Const

namespace Jarvis {
struct Point2d {
  double x, y;

  bool operator==(const Point2d& other) const { return x == other.x && y == other.y; }
};

std::vector<Point2d> convexHull(const std::vector<Point2d>& points);

class JarvisTestTaskSequential : public ppc::core::Task {
 public:
  explicit JarvisTestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  uint32_t points_count{};
  std::vector<Point2d> points{}, convex_hull{};
};
}  // namespace Jarvis
