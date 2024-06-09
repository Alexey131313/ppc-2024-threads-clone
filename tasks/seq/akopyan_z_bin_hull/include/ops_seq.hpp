// Copyright 2024 Akopyan Zal
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace akopyan_z_bin_hull {
struct pnt {
  int x{};
  int y{};
  pnt() = default;
  pnt(int x_, int y_) : x(x_), y(y_) {}
};

const int SEPARATOR = -1;

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> bin_image{};
  std::vector<int> out_hull{};
  int width{};
  int height{};

  std::vector<std::vector<pnt>> label_components();
  static std::vector<pnt> graham(std::vector<pnt> points);
};
}  // namespace akopyan_z_bin_hull
