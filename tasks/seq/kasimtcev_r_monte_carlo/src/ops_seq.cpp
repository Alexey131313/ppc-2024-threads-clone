// Copyright 2024 Kasimtcev Roman
#include "seq/kasimtcev_r_monte_carlo/include/ops_seq.hpp"

#include <cmath>
#include <thread>

using namespace std::chrono_literals;

double flinear(double x, double y) { return x + y; }
double fsinxsiny(double x, double y) { return sin(x) + sin(y); }
double fcosxcosy(double x, double y) { return cos(x) + cos(y); }
double fxy(double x, double y) { return x * y; }
double fxyy(double x, double y) { return x * y * y; }

bool MonteCarloSequentialKasimtcev::pre_processing() {
  internal_order_test();
  Int1[0] = reinterpret_cast<double*>(taskData->inputs[0])[0];
  Int1[1] = reinterpret_cast<double*>(taskData->inputs[0])[1];
  Int2[0] = reinterpret_cast<double*>(taskData->inputs[1])[0];
  Int2[1] = reinterpret_cast<double*>(taskData->inputs[1])[1];
  function = reinterpret_cast<func>(taskData->inputs[2]);

  N = 100;
  res = 0.0;
  return true;
}

bool MonteCarloSequentialKasimtcev::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 2 && taskData->inputs_count[1] == 2 && taskData->outputs_count[0] == 1;
}

bool MonteCarloSequentialKasimtcev::run() {
  internal_order_test();

  double h1 = (Int1[1] - Int1[0]) / N;
  double h2 = (Int2[1] - Int2[0]) / N;

  int i;
  int j;
  for (j = 0; j < N; ++j) {
    double y = Int2[0] + h2 * j;
    for (i = 0; i < N; ++i) {
      res += function(Int1[0] + h1 * i, y);
    }
  }
  res *= h1 * h2;

  return true;
}

bool MonteCarloSequentialKasimtcev::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}
