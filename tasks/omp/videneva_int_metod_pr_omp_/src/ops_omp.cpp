// Copyright 2024 Videneva Ekaterina
#include "omp/videneva_int_metod_pr_omp_/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <iostream>

auto g = [](double x, double y) { return std::pow(x, 2) + y; };

bool Videneva_opm_task::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 5 && taskData->outputs_count[0] == 2;
}

bool Videneva_opm_task::pre_processing() {
  internal_order_test();
  auto* inp = reinterpret_cast<double*>(taskData->inputs[0]);
  xLimL = inp[0], xLimU = inp[1], yLimL = inp[2], yLimU = inp[3], number = inp[4];
  result = 0, error = 0;
  return true;
}

bool Videneva_opm_task::run() {
  internal_order_test();
  try {
    hX = (xLimU - xLimL) / number, hY = (yLimU - yLimL) / number;
    double result_ = .0;
#pragma omp parallel for schedule(static, 1) reduction(+ : result_)
    for (int64_t i = 0; i < number; i++) {
      double pre_result = .0;
#pragma omp parallel for reduction(+ : pre_result)
      for (int64_t j = 0; j < number; j++) {
        pre_result += g(xLimL + hX * (i + 0.5), yLimL + hY * (j + 0.5));
      }
      result_ += pre_result;
    }
    result = result_ * hX * hY;
    error = 2 * (xLimU - xLimL) * (yLimU - yLimL) * (hX * hX + hY * hY) / 24;
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

bool Videneva_opm_task::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  reinterpret_cast<double*>(taskData->outputs[0])[1] = error;
  return true;
}