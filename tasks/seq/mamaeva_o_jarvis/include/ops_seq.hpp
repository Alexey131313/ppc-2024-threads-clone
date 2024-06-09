// Copyright 2024 Mamaeva Olga
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace jarvis {
struct r {
  double x, y;
  r operator-(const r& other) const { return {x - other.x, y - other.y}; }
  double operator^(const r& other) const { return x * other.y - y * other.x; }
  bool operator<(const r& other) const { return x < other.x || (x == other.x && y < other.y); }
  bool operator==(const r& other) const { return x == other.x && y == other.y; }
};
void prepare_points(r* points, int n, r* hull, int h, double r = 1.0, unsigned int seed = 42u);
}  // namespace jarvis

class JarvisHullSequential : public ppc::core::Task {
 public:
  explicit JarvisHullSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<jarvis::r> points{};
  jarvis::r* hull{};
  int hull_sz{};
};
