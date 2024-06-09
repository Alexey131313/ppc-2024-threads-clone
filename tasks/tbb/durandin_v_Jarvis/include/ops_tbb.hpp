// Copyright 2024 Durandin Vladimir
#ifndef TASKS_DURANDIN_V_JARVIS_TEST_TBB_OPS_TBB_HPP_
#define TASKS_DURANDIN_V_JARVIS_TEST_TBB_OPS_TBB_HPP_

#include <tbb/tbb.h>

#include <algorithm>
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

std::vector<Point2d> convexHullSeq(const std::vector<Point2d>& points);
std::vector<Point2d> convexHullTBB(const std::vector<Point2d>& points);

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

class JarvisTestTaskParallel : public ppc::core::Task {
 public:
  explicit JarvisTestTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  uint32_t points_count{};
  std::vector<Point2d> points{}, convex_hull{};
};
}  // namespace Jarvis

#endif  // TASKS_DURANDIN_V_JARVIS_TEST_TBB_OPS_TBB_HPP_