// Copyright 2023 Kulikov Artem
#include "tbb/kulikov_a_rect_integr/include/ops_tbb.hpp"

#include <oneapi/tbb.h>

#include <cmath>
#include <iostream>

using namespace std::chrono_literals;

auto my_f = [](double x, double y) { return std::pow(x, 2) + y; };

bool KulikovTaskTBB::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 5 && taskData->outputs_count[0] == 2;
}

bool KulikovTaskTBB::pre_processing() {
  internal_order_test();
  auto* inp = reinterpret_cast<double*>(taskData->inputs[0]);
  x_lim_l = inp[0], x_lim_u = inp[1], y_lim_l = inp[2], y_lim_u = inp[3], n = inp[4];
  res = 0, err = 0;
  return true;
}

bool KulikovTaskTBB::run() {
  internal_order_test();
  try {
    h_x = (x_lim_u - x_lim_l) / n, h_y = (y_lim_u - y_lim_l) / n;
    double sum = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, n), 0.0,
        [&](const tbb::blocked_range<size_t>& r, double sum) {
          size_t end = r.end();
          for (size_t i = r.begin(); i != end; ++i) {
            for (size_t j = 0; j < n; ++j) {
              sum += my_f(x_lim_l + h_x * (i + 0.5), y_lim_l + h_y * (j + 0.5));
            }
          }
          return sum;
        },
        std::plus<>());
    res = sum * h_x * h_y;
    err = 2 * (x_lim_u - x_lim_l) * (y_lim_u - y_lim_l) * (h_x * h_x + h_y * h_y) / 24;
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

bool KulikovTaskTBB::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  reinterpret_cast<double*>(taskData->outputs[0])[1] = err;
  return true;
}
