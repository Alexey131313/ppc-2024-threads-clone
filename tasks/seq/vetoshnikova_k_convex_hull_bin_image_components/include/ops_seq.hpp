// Copyright 2024 Vetoshnikova Ekaterina
#pragma once

#include <algorithm>
#include <stack>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class ConstructingConvexHullSeq : public ppc::core::Task {
 public:
  explicit ConstructingConvexHullSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int h{};
  int w{};
  int numComponents;
  std::vector<std::vector<uint8_t>> img = {};
  std::vector<std::vector<int>> imgMark = {};
  std::vector<int> hull = {};

  void markingComponent();
  void convexHull(int label);
};

struct Point {
  int x, y;
};
