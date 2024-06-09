// Copyright 2024 Ivlev Alexander
#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace ivlev_a_seq {
class ConvexHullSequential : public ppc::core::Task {
 public:
  explicit ConvexHullSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::pair<size_t, size_t>> sizes;
  std::vector<std::vector<std::pair<size_t, size_t>>> components;
  std::vector<std::vector<std::pair<size_t, size_t>>> results;

  static std::vector<std::pair<size_t, size_t>> Convex_Hull(const std::vector<std::pair<size_t, size_t>>& component_);
};

size_t rotation(const std::pair<int, int>& a, const std::pair<int, int>& b, const std::pair<int, int>& c);
}  // namespace ivlev_a_seq
