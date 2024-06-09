// Copyright 2024 Prokofev Kirill
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

struct Point {
  int x, y;
  Point(int x, int y) : x(x), y(y) {}
};

std::vector<int> FindComponents(const std::vector<std::vector<int>>& image, int width, int height);
void FloodFill(std::vector<int>* image, int height, int width, int yStart, int xStart, int label);
int FindCountComponents(const std::vector<int>& image);
int FindCountPointsInComponent(const std::vector<int>& image);
std::vector<int> RemoveExtraPoints(const std::vector<int>& image, int width, int height, int label);
void Sort(std::vector<int>* points, int xMin, int yMin);
std::vector<int> GrahamAlgorithm(std::vector<int> points);

class BinaryImageConvexHullSeq : public ppc::core::Task {
 public:
  explicit BinaryImageConvexHullSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>>* img{};
  std::vector<std::vector<int>> img_;
  std::vector<int> res{};
  int width, height;
};
