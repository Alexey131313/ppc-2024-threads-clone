// Copyright 2024 Kudinov Nikita
#include "seq/kudinov_n_image_filter_gauss_vertical/include/image_filter_gauss_vertical_task.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

GaussKernel::GaussKernel() : _radius(), _sigma(), _size() {}

GaussKernel::GaussKernel(std::size_t radius, double sigma) : _radius(radius), _sigma(sigma), _size(radius * 2 + 1) {
  this->_data = std::vector(this->_size * this->_size, 0.0);

  double sum = 0.0;
  for (std::size_t i = 0; i < this->_size; i += 1) {
    for (std::size_t j = 0; j < this->_size; j += 1) {
      double dist_x = std::pow(static_cast<double>(i) - static_cast<double>(this->_radius), 2.0);
      double dist_y = std::pow(static_cast<double>(j) - static_cast<double>(this->_radius), 2.0);

      double coef = std::exp(-(dist_x + dist_y) / (2.0 * std::pow(this->_sigma, 2.0)));
      this->_data[i * this->_size + j] = coef;
      sum += coef;
    }
  }

  for (std::size_t i = 0; i < this->_size; i += 1) {
    for (std::size_t j = 0; j < this->_size; j += 1) {
      this->_data[i * this->_size + j] /= sum;
    }
  }
}

std::size_t GaussKernel::radius() const { return this->_radius; }

std::size_t GaussKernel::size() const { return this->_size; }

double GaussKernel::sigma() const { return this->_sigma; }

double GaussKernel::get(std::size_t y, std::size_t x) const {
  if (y >= this->_size || x >= this->_size) {
    throw std::runtime_error("Out of bounds");
  }

  return this->_data[y * this->_size + x];
}

Image::Image() : _height(), _width() {}
Image::Image(std::size_t height, std::size_t width, const std::vector<Pixel>& pixels)
    : _height(height), _width(width), _pixels(pixels) {}

Pixel Image::get_pixel(std::size_t y, std::size_t x) const {
  if (y >= this->_height || x >= this->_width) {
    throw std::runtime_error("Pixel is out of bounds");
  }

  return this->_pixels[y * this->_width + x];
}

void Image::set_pixel(std::size_t y, std::size_t x, Pixel pixel) {
  if (y >= this->_height || x >= this->_width) {
    throw std::runtime_error("Pixel is out of bounds");
  }

  this->_pixels[y * this->_width + x] = pixel;
}

Pixel Image::_get_gauss_filtered_pixel(std::size_t y, std::size_t x, const GaussKernel& gauss_kernel) const {
  double result_pixel_value = 0;

  for (std::size_t i = 0; i < gauss_kernel.size(); i += 1) {
    for (std::size_t j = 0; j < gauss_kernel.size(); j += 1) {
      Pixel pixel = this->get_pixel(
          static_cast<std::size_t>(std::clamp<double>(
              static_cast<double>(y) - static_cast<double>(gauss_kernel.radius()) + static_cast<double>(i), 0,
              this->_height - 1)),
          static_cast<std::size_t>(std::clamp<double>(
              static_cast<double>(x) - static_cast<double>(gauss_kernel.radius()) + static_cast<double>(j), 0,
              this->_width - 1)));

      result_pixel_value +=
          std::clamp<double>(static_cast<double>(gauss_kernel.get(i, j)) * static_cast<double>(pixel), 0, UINT8_MAX);
    }
  }

  return static_cast<Pixel>(std::clamp<double>(result_pixel_value, 0, UINT8_MAX));
}

Image Image::gauss_filtered(const GaussKernel& gauss_kernel) const {
  Image out(this->_height, this->_width, std::vector<Pixel>(this->_height * this->_width, 0));

  for (std::size_t y = 0; y < this->_height; y += 1) {
    for (std::size_t x = 0; x < this->_width; x += 1) {
      out.set_pixel(y, x, this->_get_gauss_filtered_pixel(y, x, gauss_kernel));
    }
  }

  return out;
}

std::size_t Image::height() const { return this->_height; }

std::size_t Image::width() const { return this->_width; }

const std::vector<Pixel>& Image::pixels() const { return this->_pixels; }

std::size_t Image::hash() const {
  return std::hash<std::string>{}(std::string(this->_pixels.begin(), this->_pixels.end()));
}

bool ImageFilterGaussVerticalTask::validation() {
  internal_order_test();

  this->_gauss_kernel = *reinterpret_cast<GaussKernel*>(this->taskData->inputs[0]);
  this->_input_image = *reinterpret_cast<Image*>(this->taskData->inputs[1]);
  this->_output_image = *reinterpret_cast<Image*>(this->taskData->outputs[0]);

  if (this->_gauss_kernel.sigma() < 0.0) {
    return false;
  }

  if (this->_input_image.height() * this->_input_image.width() != this->_input_image.pixels().size()) {
    return false;
  }

  if (this->_input_image.height() != this->_output_image.height()) {
    return false;
  }

  if (this->_input_image.width() != this->_output_image.width()) {
    return false;
  }

  if (this->_input_image.pixels().size() != this->_output_image.pixels().size()) {
    return false;
  }

  return true;
}

bool ImageFilterGaussVerticalTask::pre_processing() {
  internal_order_test();

  return true;
}

bool ImageFilterGaussVerticalTask::run() {
  internal_order_test();

  this->_output_image = this->_input_image.gauss_filtered(this->_gauss_kernel);

  return true;
}

bool ImageFilterGaussVerticalTask::post_processing() {
  internal_order_test();

  *reinterpret_cast<Image*>(this->taskData->outputs[0]) = this->_output_image;

  return true;
}
