// Copyright 2024 Videneva Ekaterina
#include "tbb/videneva_e_integral/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <cmath>
#include <iostream>

auto videneva_func_g = [](double x, double y) { return std::pow(x, 2) + y; };

namespace Videneva_e_tbb_integral {

bool VidenevaEIntegralTBB::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 5 && taskData->outputs_count[0] == 2;
}

bool VidenevaEIntegralTBB::pre_processing() {
  internal_order_test();
  auto *inp = reinterpret_cast<double *>(taskData->inputs[0]);
  xLimL = inp[0], xLimU = inp[1], yLimL = inp[2], yLimU = inp[3], number = inp[4];
  result = 0, error = 0;
  return true;
}

bool VidenevaEIntegralTBB::run() {
  internal_order_test();
  double x_i = (xLimU >= xLimL) ? (xLimU - xLimL) / number : (xLimL - xLimU) / number;
  double y_i = (yLimU >= yLimL) ? (yLimU - yLimL) / number : (yLimL - yLimU) / number;

  try {
    result = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, number), 0.0,
        [&](const tbb::blocked_range<int> &r, double local_res) {
          for (int i = r.begin(); i != r.end(); ++i) {
            for (int j = 0; j < number; ++j) {
              local_res += videneva_func_g(xLimL + x_i * (i + 0.5), yLimL + y_i * (j + 0.5));
            }
          }
          return local_res;
        },
        std::plus<>());
    result *= x_i * y_i;
    error = 2 * (xLimU - xLimL) * (yLimU - yLimL) * (x_i * x_i + y_i * y_i) / 24;
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

bool VidenevaEIntegralTBB::post_processing() {
  internal_order_test();
  reinterpret_cast<double *>(taskData->outputs[0])[0] = result;
  reinterpret_cast<double *>(taskData->outputs[0])[1] = error;
  return true;
}

}  // namespace Videneva_e_tbb_integral