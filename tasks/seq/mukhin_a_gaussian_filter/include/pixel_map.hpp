// Copyright 2024 Mukhin Ivan
#pragma once

#include <sys/types.h>

#include <cstdint>
#include <vector>

class Pixel {
 public:
  uint8_t r, g, b;

  friend bool operator==(const Pixel& a, const Pixel& b);

  friend bool operator!=(const Pixel& a, const Pixel& b) { return !(a == b); }
};

class PixelMap {
 private:
  uint32_t n;
  uint32_t m;

  void set_data(uint8_t r, uint8_t g, uint8_t b);

 public:
  std::vector<Pixel> data;
  PixelMap() : n(0), m(0){};

  PixelMap(uint32_t n_, uint32_t m_, uint8_t r = 0, uint8_t g = 0, uint8_t b = 0);
  uint32_t width() const { return n; }
  uint32_t height() const { return m; }
  ~PixelMap() = default;

  Pixel& get_pixel(uint32_t i, uint32_t j);

  const Pixel& get_pixel(uint32_t i, uint32_t j) const;

  friend bool operator==(const PixelMap& a, const PixelMap& b);

  friend bool operator!=(const PixelMap& a, const PixelMap& b) { return !(a == b); }
};
