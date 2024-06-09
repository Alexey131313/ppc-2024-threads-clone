// Copyright 2024 Kozlov Mikhail
#include "omp/kozlov_m_simpson_integral/include/ops_omp.hpp"

double kozlov_omp::xy(double x, double y) { return x * y; }
double kozlov_omp::sinxy(double x, double y) { return std::sin(x * y); }
double kozlov_omp::linear(double x, double y) { return y * 5 - x * 2; }
double kozlov_omp::expxy(double x, double y) { return std::exp(x * y); }
double kozlov_omp::expy_x(double x, double y) { return std::exp(y - x) / 2; }

bool kozlov_omp::KozlovTasknOmpSequential::pre_processing() {
  internal_order_test();
  auto* tmp = reinterpret_cast<double*>(taskData->inputs[0]);
  x1 = tmp[0], x2 = tmp[1], y1 = tmp[2], y2 = tmp[3];
  res = 0.0;
  f = reinterpret_cast<FUNC>(taskData->inputs[1]);
  n = reinterpret_cast<uint64_t*>(taskData->inputs[2])[0];
  m = reinterpret_cast<uint64_t*>(taskData->inputs[3])[0];
  return true;
}

bool kozlov_omp::KozlovTasknOmpSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 4 && taskData->outputs_count[0] == 1;
}

bool kozlov_omp::KozlovTasknOmpSequential::run() {
  internal_order_test();
  double h_x = std::abs(x2 - x1) / n;
  double h_y = std::abs(y2 - y1) / m;
  double q;
  double p;
  double x;
  double y;
  double res_ = 0;
  for (int i = 0; i <= n; i++) {
    if (i == 0 || i == n) {
      p = 1;
    } else if (i % 2 == 0) {
      p = 4;
    } else {
      p = 2;
    }
    for (int j = 0; j <= m; j++) {
      if (j == 0 || j == m) {
        q = 1;
      } else if (j % 2 == 0) {
        q = 4;
      } else {
        q = 2;
      }
      x = x1 + i * h_x;
      y = y1 + j * h_y;
      res_ += p * q * f(x, y);
    }
  }

  res_ *= h_x * h_y / 9;
  res = res_;
  return true;
}

bool kozlov_omp::KozlovTasknOmpSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

bool kozlov_omp::KozlovTasknOmpParallel::pre_processing() {
  internal_order_test();
  auto* tmp = reinterpret_cast<double*>(taskData->inputs[0]);
  x1 = tmp[0], x2 = tmp[1], y1 = tmp[2], y2 = tmp[3];
  res = 0.0;
  f = reinterpret_cast<FUNC>(taskData->inputs[1]);
  n = reinterpret_cast<uint64_t*>(taskData->inputs[2])[0];
  m = reinterpret_cast<uint64_t*>(taskData->inputs[3])[0];
  return true;
}

bool kozlov_omp::KozlovTasknOmpParallel::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 4 && taskData->outputs_count[0] == 1;
}

bool kozlov_omp::KozlovTasknOmpParallel::run() {
  internal_order_test();
  double h_x = std::abs(x2 - x1) / n;
  double h_y = std::abs(y2 - y1) / m;
  double q;
  double p;
  double x;
  double y;
  double res_ = 0;
#pragma omp parallel for reduction(+ : res_)
  for (int i = 0; i <= n; i++) {
    if (i == 0 || i == n) {
      p = 1;
    } else if (i % 2 == 0) {
      p = 4;
    } else {
      p = 2;
    }
    for (int j = 0; j <= m; j++) {
      if (j == 0 || j == m) {
        q = 1;
      } else if (j % 2 == 0) {
        q = 4;
      } else {
        q = 2;
      }
      x = x1 + i * h_x;
      y = y1 + j * h_y;
      res_ += p * q * f(x, y);
    }
  }

  res_ *= h_x * h_y / 9;
  res = res_;
  return true;
}

bool kozlov_omp::KozlovTasknOmpParallel::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}
