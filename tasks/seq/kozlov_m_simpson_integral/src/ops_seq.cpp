// Copyright 2024 Kozlov Mikhail
#include "seq/kozlov_m_simpson_integral/include/ops_seq.hpp"

double xy(double x, double y) { return x * y; }
double sinxy(double x, double y) { return std::sin(x * y); }
double linear(double x, double y) { return y * 5 - x * 2; }
double expxy(double x, double y) { return std::exp(x * y); }
double expy_x(double x, double y) { return std::exp(y - x) / 2; }

bool KozlovTaskSequential::pre_processing() {
  internal_order_test();
  auto* tmp = reinterpret_cast<double*>(taskData->inputs[0]);
  x1 = tmp[0], x2 = tmp[1], y1 = tmp[2], y2 = tmp[3];
  res = 0.0;
  f = reinterpret_cast<FUNC>(taskData->inputs[1]);
  n = reinterpret_cast<uint64_t*>(taskData->inputs[2])[0];
  m = reinterpret_cast<uint64_t*>(taskData->inputs[3])[0];
  return true;
}

bool KozlovTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 4 && taskData->outputs_count[0] == 1;
}

bool KozlovTaskSequential::run() {
  internal_order_test();
  double h_x = std::abs(x2 - x1) / n;
  double h_y = std::abs(y2 - y1) / m;
  double q;
  double p;
  double x;
  double y;
  for (uint64_t i = 0; i <= n; i++) {
    if (i == 0 || i == n) {
      p = 1;
    } else if (i % 2 == 0) {
      p = 4;
    } else {
      p = 2;
    }
    for (uint64_t j = 0; j <= m; j++) {
      if (j == 0 || j == m) {
        q = 1;
      } else if (j % 2 == 0) {
        q = 4;
      } else {
        q = 2;
      }
      x = x1 + i * h_x;
      y = y1 + j * h_y;
      res += p * q * f(x, y);
    }
  }

  res *= h_x * h_y / 9;
  return true;
}

bool KozlovTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}