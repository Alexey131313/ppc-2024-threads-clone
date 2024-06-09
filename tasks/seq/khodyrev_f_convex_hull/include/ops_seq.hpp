// Copyright 2024 Khodyrev Fedor
#pragma once

#include <memory>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

struct Pixel {
  int x, y;
};

bool isWhite(const std::vector<int>& image, int width, int x, int y);

class KhodyrevTaskSequential : public ppc::core::Task {
 public:
  explicit KhodyrevTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int width_in{};
  int height_in{};
  int width_out{};
  int height_out{};
  std::vector<Pixel> pixels;
  std::vector<int> image;
  std::stack<Pixel> hull;
  Pixel start_point;
  std::vector<int> convexHull;
  std::vector<int> result;
};
