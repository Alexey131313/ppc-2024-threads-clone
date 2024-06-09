// Copyright 2024 Ivanov Nikita
#include "tbb/ivanov_n_int_simpson/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <cmath>
#include <numeric>
#include <thread>

using namespace std::chrono_literals;

double linear_fun(double x, double y) { return x + y; }

double sin_cos(double x, double y) { return sin(x) + cos(y); }

double x_mul_y(double x, double y) { return x * y; }

double simpson(double x0, double x1, double y, func f) { return f(x0, y) + 4 * f((x0 + x1) / 2, y) + f(x1, y); }

bool TestTBBTaskSequentialIvanovSimpson::pre_processing() {
  internal_order_test();
  // Init value for input and output
  a = reinterpret_cast<int*>(taskData->inputs[0])[0];
  b = reinterpret_cast<int*>(taskData->inputs[0])[1];
  c = reinterpret_cast<int*>(taskData->inputs[0])[2];
  d = reinterpret_cast<int*>(taskData->inputs[0])[3];
  n = reinterpret_cast<int*>(taskData->inputs[0])[4];
  res = 0.0;
  return true;
}

bool TestTBBTaskSequentialIvanovSimpson::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 5 && taskData->outputs_count[0] == 1;
}

bool TestTBBTaskSequentialIvanovSimpson::run() {
  internal_order_test();

  double h1 = (double)(b - a) / n;
  double h2 = (double)(d - c) / n;

  for (int i = 0; i < n; i++) {
    double y0 = c + i * h2;
    double y1 = c + (i + 1) * h2;
    for (int j = 0; j < n; j++) {
      double x0 = a + j * h1;
      double x1 = a + (j + 1) * h1;

      res += h2 * h1 / 36 *
             (simpson(x0, x1, y0, fun) + 4 * simpson(x0, x1, (y0 + y1) / 2, fun) + simpson(x0, x1, y1, fun));
    }
  }

  return true;
}

bool TestTBBTaskSequentialIvanovSimpson::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

bool TestTBBTaskParallelIvanovSimpson::pre_processing() {
  internal_order_test();
  // Init value for input and output
  a = reinterpret_cast<int*>(taskData->inputs[0])[0];
  b = reinterpret_cast<int*>(taskData->inputs[0])[1];
  c = reinterpret_cast<int*>(taskData->inputs[0])[2];
  d = reinterpret_cast<int*>(taskData->inputs[0])[3];
  n = reinterpret_cast<int*>(taskData->inputs[0])[4];
  res = 0.0;
  return true;
}

bool TestTBBTaskParallelIvanovSimpson::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 5 && taskData->outputs_count[0] == 1;
}

bool TestTBBTaskParallelIvanovSimpson::run() {
  internal_order_test();

  double h1 = (double)(b - a) / n;
  double h2 = (double)(d - c) / n;

  res = oneapi::tbb::parallel_reduce(
      oneapi::tbb::blocked_range<int>(0, n), 0.0,
      [&](const tbb::blocked_range<int>& r, double total) {
        for (int i = r.begin(); i != r.end(); i++) {
          double y0 = c + i * h2;
          double y1 = c + (i + 1) * h2;
          for (int j = 0; j < n; j++) {
            double x0 = a + j * h1;
            double x1 = a + (j + 1) * h1;

            total += h2 * h1 / 36 *
                     (simpson(x0, x1, y0, fun) + 4 * simpson(x0, x1, (y0 + y1) / 2, fun) + simpson(x0, x1, y1, fun));
          }
        }
        return total;
      },
      std::plus<>());

  return true;
}

bool TestTBBTaskParallelIvanovSimpson::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}