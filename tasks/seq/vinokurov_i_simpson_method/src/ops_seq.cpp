// Copyright 2024 Vinokurov Ivan
#include "seq/vinokurov_i_simpson_method/include/ops_seq.hpp"

#include <cmath>
#include <thread>

using namespace std::chrono_literals;

double fn_simpson(func _fn, double _x0, double _x1, double _y) {
  return _fn(_x0, _y) + 4 * _fn((_x0 + _x1) / 2, _y) + _fn(_x1, _y);
}

double fn_linear(double _x, double _y) { return 9 * _x - _y; }

double fn_exp(double _x, double _y) { return exp(_x - _y); }

double fn_trig(double _x, double _y) { return sin(_x * _y); }

double fn_mul(double _x, double _y) { return _x * _x * _y; }

double fn_other(double _x, double _y) { return (_x + _y) * _y; }

bool SimpsonMethodSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  auto* inputData = reinterpret_cast<int*>(taskData->inputs[0]);

  a = inputData[0];
  b = inputData[1];
  c = inputData[2];
  d = inputData[3];
  n = inputData[4];

  result = 0.0;

  return true;
}

bool SimpsonMethodSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 5 && taskData->outputs_count[0] == 1;
}

bool SimpsonMethodSequential::run() {
  internal_order_test();

  double part1 = static_cast<double>(b - a) / n;
  double part2 = static_cast<double>(d - c) / n;

  for (int i = 0; i < n; i++) {
    double a1 = c + i * part2;
    double a2 = c + (i + 1) * part2;
    for (int j = 0; j < n; j++) {
      double b1 = a + j * part1;
      double b2 = a + (j + 1) * part1;

      result += (part1 * part2 / 36) *
                (fn_simpson(fn, b1, b2, a1) + 4 * fn_simpson(fn, b1, b2, (a1 + a2) / 2) + fn_simpson(fn, b1, b2, a2));
    }
  }

  return true;
}

bool SimpsonMethodSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  return true;
}
