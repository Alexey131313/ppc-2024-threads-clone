// Copyright 2024 Mukhin Ivan
#include "seq/mukhin_a_gaussian_filter/include/gaussian_filter.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

#include "seq/mukhin_a_gaussian_filter/include/pixel_map.hpp"

bool GaussianFilterSeq::pre_processing() {
  internal_order_test();
  input = taskData->inputs[0];
  output = taskData->outputs[0];
  width_input = taskData->inputs_count[0];
  height_input = taskData->inputs_count[1];
  width_out = taskData->outputs_count[0];
  height_out = taskData->outputs_count[1];
  image.data.resize(width_input * height_input);
  std::memcpy(image.data.data(), input, width_input * height_input * 3);
  return true;
}

bool GaussianFilterSeq::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 3 && taskData->inputs_count[1] >= 3 &&
         taskData->outputs_count[0] == taskData->inputs_count[0] &&
         taskData->outputs_count[1] == taskData->inputs_count[1];
}

bool GaussianFilterSeq::post_processing() {
  internal_order_test();
  std::memcpy(output, image.data.data(), width_input * height_input * 3);
  return true;
}

bool GaussianFilterSeq::run() {
  internal_order_test();
  filter_to_image();
  return true;
}

void GaussianFilterSeq::filter_to_image() {
  for (uint32_t i = 0; i < width_input; i++) {
    for (uint32_t j = 0; j < height_input; j++) {
      image.get_pixel(i, j) = get_new_pixel(i, j);
    }
  }
}

Pixel GaussianFilterSeq::get_new_pixel(uint32_t w, uint32_t h) {
  double result_r = 0;
  double result_b = 0;
  double result_g = 0;
  for (int i = -rad; i <= rad; i++) {
    for (int j = -rad; j <= rad; j++) {
      uint32_t new_h = h + j;
      uint32_t new_w = w + i;
      new_h = clamp(new_h, width_input);
      Pixel neighborColor = image.get_pixel(new_w, new_h);
      result_r += neighborColor.r * kernel[i + rad][j + rad];
      result_b += neighborColor.b * kernel[i + rad][j + rad];
      result_g += neighborColor.g * kernel[i + rad][j + rad];
    }
  }
  return Pixel({(uint8_t)std::round(result_r), (uint8_t)std::round(result_g), (uint8_t)std::round(result_b)});
}

uint32_t GaussianFilterSeq::clamp(uint32_t value, uint32_t max) {
  if (value < 0) return 0;
  if (value >= max) return max - 1;
  return value;
}

void GaussianFilterSeq::create_gaussian_kernel() {
  double sigm = 2.0;
  double norm = 0.0;
  for (int i = -rad; i <= rad; i++) {
    for (int j = -rad; j <= rad; j++) {
      kernel[i + rad][j + rad] = (std::exp(-(i * i + j * j) / (2 * sigm * sigm)));
      norm += kernel[i + rad][j + rad];
    }
  }
  for (uint32_t i = 0; i < kern_size; i++) {
    for (uint32_t j = 0; j < kern_size; j++) {
      kernel[i][j] /= norm;
    }
  }
}
