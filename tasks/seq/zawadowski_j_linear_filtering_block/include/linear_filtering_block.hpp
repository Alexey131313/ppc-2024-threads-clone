// Copyright 2024 Zawadowski Jan
#pragma once

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

class LinearFiltering : public ppc::core::Task {
 public:
  explicit LinearFiltering(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void sizeSet(uint16_t _width, uint16_t _height);
  void blockSet(uint16_t _width, uint16_t _height);
  void imageSet(std::vector<uint8_t> _image);
  void kernelSet(const std::vector<int16_t> &kernel);
  void getRandomImage();
  void applyGaussianFilter();

 private:
  std::vector<int16_t> gaussKernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  std::vector<uint8_t> image;
  uint16_t width = 3;
  uint16_t height = 3;
  uint16_t widthOut;
  uint16_t heightOut;
  uint16_t blockWidth = 3;
  uint16_t blockHeight = 3;
  uint8_t *input;
  uint8_t *output;
};
