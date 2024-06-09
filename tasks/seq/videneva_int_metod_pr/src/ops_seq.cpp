// Copyright 2023 Videneva Ekaterina
#include "seq/videneva_int_metod_pr/include/ops_seq.hpp"

#include <cmath>
#include <iostream>

using namespace std::chrono_literals;

auto g = [](double x, double y) { return std::pow(x, 2) + y; };

double funcLin(double x, double y) { return x - y; }
double f_x_y(double x, double y) { return x * y; }

bool Videneva_seq_task::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 5 && taskData->outputs_count[0] == 2;
}

bool Videneva_seq_task::run() {
  internal_order_test();
  try {
    hX = (xLimU - xLimL) / number, hY = (yLimU - yLimL) / number;
    for (uint64_t i = 0; i < number; i++) {
      for (uint64_t j = 0; j < number; j++) {
        result += g(xLimL + hX * (i + 0.5), yLimL + hY * (j + 0.5));
      }
    }
    result *= hX * hY;
    error = 2 * (xLimU - xLimL) * (yLimU - yLimL) * (hX * hX + hY * hY) / 24;
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

bool Videneva_seq_task::pre_processing() {
  internal_order_test();
  auto* inp = reinterpret_cast<double*>(taskData->inputs[0]);
  xLimL = inp[0], xLimU = inp[1], yLimL = inp[2], yLimU = inp[3], number = inp[4];
  result = 0, error = 0;
  return true;
}

bool Videneva_seq_task::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  reinterpret_cast<double*>(taskData->outputs[0])[1] = error;
  return true;
}