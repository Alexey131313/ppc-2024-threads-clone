// Copyright 2024 Mukhin Ivan
#pragma once
#include <sys/types.h>

#include <cstdint>

#include "core/task/include/task.hpp"
#include "omp/mukhin_a_gaussian_filter/include/pixel_map.hpp"

namespace mukhin_i_omp {
class GaussianFilterOMP : public ppc::core::Task {
 public:
  explicit GaussianFilterOMP(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  static const uint32_t kern_size = 3;
  int rad = kern_size / 2;
  double kernel[kern_size][kern_size] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  uint8_t *input, *output;
  uint32_t width_input, height_input, width_out, height_out;
  PixelMap image;

  void create_gaussian_kernel();

  static uint32_t clamp(uint32_t value, uint32_t max);

  void filter_to_image();

  Pixel get_new_pixel(uint32_t w, uint32_t h);
};
}  // namespace mukhin_i_omp
