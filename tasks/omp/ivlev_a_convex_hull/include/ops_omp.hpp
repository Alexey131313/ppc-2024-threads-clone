// Copyright 2024 Ivlev Alexander
#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace ivlev_a_omp {
class ConvexHullOMPTaskParallel : public ppc::core::Task {
 public:
  explicit ConvexHullOMPTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::pair<size_t, size_t>> sizes;
  std::vector<std::vector<std::pair<size_t, size_t>>> components;
  std::vector<std::vector<std::pair<size_t, size_t>>> results;

  static std::vector<std::pair<size_t, size_t>> Convex_Hull_tmp(
      size_t p, size_t last, size_t n, const std::vector<std::pair<size_t, size_t>>& component_);
  static std::vector<std::pair<size_t, size_t>> Convex_Hull(const std::vector<std::pair<size_t, size_t>>& component_);
};

size_t rotation(const std::pair<ptrdiff_t, ptrdiff_t>& a, const std::pair<ptrdiff_t, ptrdiff_t>& b,
                const std::pair<ptrdiff_t, ptrdiff_t>& c);
}  // namespace ivlev_a_omp
