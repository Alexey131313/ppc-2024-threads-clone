// Copyright 2023 Kulikov Artem
#include "omp/kulikov_a_rect_integr/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <iostream>

auto f = [](double x, double y) { return std::pow(x, 2) + y; };

bool KulikovTaskOMP::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 5 && taskData->outputs_count[0] == 2;
}

bool KulikovTaskOMP::pre_processing() {
  internal_order_test();
  auto* inp = reinterpret_cast<double*>(taskData->inputs[0]);
  x_lim_l = inp[0], x_lim_u = inp[1], y_lim_l = inp[2], y_lim_u = inp[3], n = inp[4];
  res = 0, err = 0;
  return true;
}

bool KulikovTaskOMP::run() {
  internal_order_test();
  try {
    h_x = (x_lim_u - x_lim_l) / n, h_y = (y_lim_u - y_lim_l) / n;
    double res_ = .0;
#pragma omp parallel for schedule(static, 1) reduction(+ : res_)
    for (int64_t i = 0; i < n; i++) {
      double pre_res = .0;
#pragma omp parallel for reduction(+ : pre_res)
      for (int64_t j = 0; j < n; j++) {
        pre_res += f(x_lim_l + h_x * (i + 0.5), y_lim_l + h_y * (j + 0.5));
      }
      res_ += pre_res;
    }
    res = res_ * h_x * h_y;
    err = 2 * (x_lim_u - x_lim_l) * (y_lim_u - y_lim_l) * (h_x * h_x + h_y * h_y) / 24;
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

bool KulikovTaskOMP::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  reinterpret_cast<double*>(taskData->outputs[0])[1] = err;
  return true;
}
