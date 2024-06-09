// Copyright 2024 Kriseev Mikhail
#pragma once

#include <vector>

#include "core/task/include/task.hpp"
namespace KriseevMTaskSeq {

using Point = std::pair<double, double>;

class ConvexHullTask : public ppc::core::Task {
 public:
  // Task data layout:
  // inputs:
  //   [0]: double * pointsX
  //   [1]: double * pointsY
  //  outputs:
  //   [0]: double * pointsX
  //   [1]: double * pointsY

  explicit ConvexHullTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<Point> points;
  std::vector<Point> finalHull;
};
}  // namespace KriseevMTaskSeq
