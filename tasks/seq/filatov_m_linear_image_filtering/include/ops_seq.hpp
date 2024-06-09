// Copyright 2024 Filatov Maxim
#pragma once

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using std::vector;

struct Color {
  uint8_t R, G, B;
  Color();
};

struct ColorF {
  float R, G, B;
  ColorF();
};

class GaussFilterHorizontal : public ppc::core::Task {
 public:
  explicit GaussFilterHorizontal(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  void initializeData();
  void copyInputToImage();
  bool validation() override;
  bool validation_is_input_size_valid();
  bool validation_is_output_size_valid_first();
  bool validation_is_output_size_valid_second();
  bool run() override;
  bool post_processing() override;
  bool copyImageData();

 private:
  uint8_t *input, *output;
  uint32_t width, height;
  vector<Color> image;

  static const size_t kSize = 3;
  double kernel[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  void makeKernel(float sigma = 2);
  void calculateGaussianValues(float sigma, float* normalizationFactor);
  void normalizeKernel(float normalizationFactor);
  void applyKernel();
  Color calculateNewPixelColor(size_t x, size_t y);
  void static calculateSingleColorComponent(uint8_t neighborColor, float kernelValue, float* newColorComponent);
  void calculateColorComponents(size_t x, size_t y, ColorF* color);
  void calculateColorComponentsForRow(int64_t l, size_t x, size_t y, ColorF* color);
  Color static calculateCeilColorF(ColorF preparedColor);
  void calculateColorsComponents(Color* neighborColor, int64_t k, int64_t l, int64_t halfSize, ColorF* color);
  template <typename T>
  T clamp(const T& val, const T& min, const T& max);
};
