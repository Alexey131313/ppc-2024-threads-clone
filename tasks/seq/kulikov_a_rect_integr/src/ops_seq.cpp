// Copyright 2023 Kulikov Artem
#include "seq/kulikov_a_rect_integr/include/ops_seq.hpp"

#include <cmath>
#include <iostream>

using namespace std::chrono_literals;

auto f = [](double x, double y) { return std::pow(x, 2) + y; };

bool KulikovTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 5 && taskData->outputs_count[0] == 2;
}

bool KulikovTaskSequential::pre_processing() {
  internal_order_test();
  auto* inp = reinterpret_cast<double*>(taskData->inputs[0]);
  x_lim_l = inp[0], x_lim_u = inp[1], y_lim_l = inp[2], y_lim_u = inp[3], n = inp[4];
  res = 0, err = 0;
  return true;
}

bool KulikovTaskSequential::run() {
  internal_order_test();
  try {
    h_x = (x_lim_u - x_lim_l) / n, h_y = (y_lim_u - y_lim_l) / n;
    for (uint64_t i = 0; i < n; i++) {
      for (uint64_t j = 0; j < n; j++) {
        res += f(x_lim_l + h_x * (i + 0.5), y_lim_l + h_y * (j + 0.5));
      }
    }
    res *= h_x * h_y;
    err = 2 * (x_lim_u - x_lim_l) * (y_lim_u - y_lim_l) * (h_x * h_x + h_y * h_y) / 24;
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

bool KulikovTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  reinterpret_cast<double*>(taskData->outputs[0])[1] = err;
  return true;
}
