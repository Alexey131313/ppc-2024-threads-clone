// Copyright 2024 Kudinov Nikita
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

using Pixel = std::uint8_t;

class GaussKernel {
 public:
  GaussKernel();
  GaussKernel(std::size_t radius, double sigma);

  std::size_t radius() const;
  std::size_t size() const;
  double sigma() const;
  double get(std::size_t y, std::size_t x) const;

 private:
  std::size_t _radius;
  double _sigma;
  std::vector<double> _data;

  std::size_t _size;
};

class Image {
 public:
  Image();
  Image(std::size_t height, std::size_t _width, const std::vector<Pixel>& _pixels);
  Pixel get_pixel(std::size_t y, std::size_t x) const;
  void set_pixel(std::size_t y, std::size_t x, Pixel pixel);

  Image gauss_filtered(const GaussKernel& gauss_kernel) const;
  std::size_t height() const;
  std::size_t width() const;
  const std::vector<Pixel>& pixels() const;
  std::size_t hash() const;

 private:
  std::size_t _height;
  std::size_t _width;
  std::vector<Pixel> _pixels;

  Pixel _get_gauss_filtered_pixel(std::size_t y, std::size_t x, const GaussKernel& gauss_kernel) const;
};

class ImageFilterGaussVerticalTask : public ppc::core::Task {
  // inputs:
  //   [0]: gauss_kernel | 1
  //   [1]: input_image | 1
  // outputs:
  //   [0]: output_image | 1
 public:
  explicit ImageFilterGaussVerticalTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  GaussKernel _gauss_kernel;
  Image _input_image;
  Image _output_image;
};
