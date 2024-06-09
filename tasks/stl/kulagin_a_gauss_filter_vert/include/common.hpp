// Copyright 2024 Kulagin Aleksandr
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace kulagin_a_gauss {
static const uint32_t default_alpha = 0x0;

inline const uint32_t& get_color(const size_t& w, const size_t&, const uint32_t* img, const size_t& px,
                                 const size_t& py) {
  return img[px + py * w];
}

inline uint32_t& get_color(const size_t& w, const size_t&, uint32_t* img, const size_t& px, const size_t& py) {
  return img[px + py * w];
}

inline uint32_t get_color_channel(const uint32_t& col, const uint32_t& i) noexcept { return (col >> (i * 8)) & (0xff); }

template <bool is_channel_clean = true>
inline void put_color_channel(uint32_t& col, const uint32_t& ch, const uint32_t& i) noexcept {
  if (!is_channel_clean) {
    col &= ~(0xff << (i * 8)) & ~(~default_alpha << 24);
  }
  col |= ((ch & 0xff) << (i * 8));
}

template <bool check_border = true>
inline void apply_filter_pixel(const size_t& w, const size_t& h, const uint32_t* img, const float* kernel,
                               uint32_t* res_img, const size_t& px, const size_t& py) {
  float res_pxl[3] = {0.0f, 0.0f, 0.0f};
  uint32_t i;
  for (size_t l = 0; l < 3; l++) {
    for (size_t k = 0; k < 3; k++) {
      size_t dx = px + k - 1;
      size_t dy = py + l - 1;
      if (check_border) {
        if (px == 0 && k == 0) {
          dx = 0;
        } else if (dx == w) {
          dx--;
        }
        if (py == 0 && l == 0) {
          dy = 0;
        } else if (dy == h) {
          dy--;
        }
      }
      const uint32_t& tmp_col = get_color(w, h, img, dx, dy);
      for (i = 0; i < 3; i++) {
        res_pxl[i] += (static_cast<float>(get_color_channel(tmp_col, i))) * kernel[k + l * 3];
      }
    }
  }
  for (i = 0; i < 3; i++) {
    put_color_channel(get_color(w, h, res_img, px, py), std::clamp<uint32_t>(res_pxl[i], 0, 255), i);
  }
}

template <bool check_border = true>
inline void apply_filter_line(const size_t& w, const size_t& h, const uint32_t* img, const float* kernel,
                              uint32_t* res_img, const size_t& x) {
  size_t y;
  apply_filter_pixel<true>(w, h, img, kernel, res_img, x, 0);
  for (y = 1; y < h - 1; y++) {
    apply_filter_pixel<check_border>(w, h, img, kernel, res_img, x, y);
  }
  if (h != 1) {
    apply_filter_pixel<true>(w, h, img, kernel, res_img, x, h - 1);
  }
}

inline void apply_filter(const size_t& w, const size_t& h, const uint32_t* img, const float* kernel,
                         uint32_t* img_res) {
  kulagin_a_gauss::apply_filter_line<true>(w, h, img, kernel, img_res, 0);
  for (size_t x = 1; x < w - 1; x++) {
    kulagin_a_gauss::apply_filter_line<false>(w, h, img, kernel, img_res, x);
  }
  if (w != 1) {
    kulagin_a_gauss::apply_filter_line<true>(w, h, img, kernel, img_res, w - 1);
  }
}

inline std::vector<uint32_t> generator1(const size_t& w, const size_t& h) {
  std::vector<uint32_t> img(w * h);
  for (size_t i = 0; i < w * h; i++) {
    uint32_t col = 0;
    for (size_t j = 0; j < 3; j++) {
      put_color_channel(col, (i * (j + 1)) % 256, j);
    }
    img[i] = col;
  }
  return img;
}

inline std::vector<float> generate_kernel() { return std::vector<float>(3 * 3, 1.0f / 9.0f); }

inline std::vector<float> generate_kernel(const float& sigma) {
  if (std::abs(sigma) <= std::numeric_limits<float>::epsilon() || std::isinf(sigma)) {
    // if sigma is 0
    return generate_kernel();
  }
  std::vector<float> kernel(3 * 3);
  float norm = 0.0f;
  for (size_t i = 0; i < 3; i++) {
    const size_t x = i % 2 + 1;
    for (size_t j = 0; j < 3; j++) {
      const size_t y = j % 2 + 1;
      kernel[i + 3 * j] = expf(-static_cast<float>(x * x + y * y) / (2.0f * sigma * sigma));
      norm += kernel[i + 3 * j];
    }
  }
  for (size_t i = 0; i < 9; i++) {
    kernel[i] /= norm;
  }
  return kernel;
}

inline std::string generate_path_to_img(const std::string& task, const std::string& task_name,
                                        const std::string& filename) {
  return (std::filesystem::path() / PATH_TO_PPC_PROJECT / "tasks" / task / task_name / "data" / filename).string();
}

inline bool get_img_from_file(const std::string& path, std::vector<uint32_t>& img, size_t& w, size_t& h) {
  std::ifstream f(path, std::ios::binary | std::ios::in);
  if (!f) {
    return false;
  }
  uint32_t _w;
  uint32_t _h;
  f.read(reinterpret_cast<char*>(&_w), sizeof(uint32_t));
  f.read(reinterpret_cast<char*>(&_h), sizeof(uint32_t));
  w = _w;
  h = _h;
  img.resize(w * h);
  for (size_t i = 0; i < w * h; i++) {
    f.read(reinterpret_cast<char*>(&img[i]), sizeof(uint32_t));
  }
  f.close();
  return f.good();
}
}  // namespace kulagin_a_gauss
