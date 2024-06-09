// Copyright 2024 Moiseev Nikita
#pragma once
#include <tbb/tbb.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

struct Point {
  int x, y;

  bool operator==(const Point& other) const { return x == other.x && y == other.y; }
  bool operator<(const Point& other) const {
    if (x != other.x) {
      return x < other.x;
    }
    return y < other.y;
  }
};

class TestTaskSequentialJarvisMoiseev : public ppc::core::Task {
 public:
  explicit TestTaskSequentialJarvisMoiseev(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<Point> points;
  std::vector<Point> convexHullPoints;
};

class TestTaskTbbJarvisMoiseev : public ppc::core::Task {
 public:
  explicit TestTaskTbbJarvisMoiseev(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<Point> points;
  std::vector<Point> convexHullPoints;
};
std::vector<Point> Jarvis_Moiseev(const std::vector<Point>& points);
std::vector<Point> tbbJarvis_Moiseev(const std::vector<Point>& points);