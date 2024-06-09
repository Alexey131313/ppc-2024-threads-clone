// Copyright 2024 Ivanchenko Aleksei
#pragma once

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

using std::vector;

struct Color {
  uint8_t R, G, B;
};

class GaussFilterSequential : public ppc::core::Task {
 public:
  explicit GaussFilterSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  uint8_t *input, *output;
  uint32_t width, height;
  vector<Color> image;

  static const size_t kernelSize = 3;
  double kernel[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  void createKernel(float sigma = 2);
  void applyKernel();
  Color calculateNewPixelColor(size_t x, size_t y);
};
