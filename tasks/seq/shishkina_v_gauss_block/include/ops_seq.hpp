// Copyright 2023 Shishkina Valeriya
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

std::vector<int> getPicture(int n, int m, uint8_t min, uint8_t max);

class LinearFilteringGauss : public ppc::core::Task {
 public:
  explicit LinearFilteringGauss(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  int getPixel(int x, int y) { return input[x * width + y]; }
  void setPixel(int x, int y, int value) { res[x * width + y] = value; }

 private:
  std::vector<int> input = {};
  std::vector<int> res = {};
  int height{}, width{};
  int min{}, max{};
};
