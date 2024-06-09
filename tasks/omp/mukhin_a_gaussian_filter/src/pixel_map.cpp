// Copyright 2024 Mukhin Ivan
#include "omp/mukhin_a_gaussian_filter/include/pixel_map.hpp"

#include <cstdint>

void PixelMap::set_data(uint8_t r, uint8_t g, uint8_t b) {
  data.resize(n * m);
  for (uint32_t i = 0; i < n * m; i++) {
    data[i].r = r;
    data[i].g = g;
    data[i].b = b;
  }
}

bool operator==(const Pixel& a, const Pixel& b) { return (a.r == b.r && a.g == b.g && a.b == b.b); }

PixelMap::PixelMap(uint32_t n_, uint32_t m_, uint8_t r, uint8_t g, uint8_t b) {
  this->n = n_;
  this->m = m_;
  set_data(r, g, b);
}

Pixel& PixelMap::get_pixel(uint32_t i, uint32_t j) { return this->data[i * width() + j]; }

const Pixel& PixelMap::get_pixel(uint32_t i, uint32_t j) const { return this->data[i * width() + j]; }

bool operator==(const PixelMap& a, const PixelMap& b) {
  if (a.width() == b.width() && a.height() == b.height()) {
    for (uint32_t i = 0; i < a.width() * a.height(); i++) {
      if (a.data[i] != b.data[i]) {
        return false;
      }
    }
    return true;
  }
  return false;
}