// Copyright 2024 Mortina Nastya
#include "omp/mortina_a_int_trapezoid_omp/include/ops_omp.hpp"

#include <cmath>
#include <functional>
#include <iostream>

namespace Mortina_a_omp_integral_trapezoid {

double trapezoidal_integral(double a1, double b1, double a2, double b2, int n1, int n2,
                            const std::function<double(double, double)> &f) {
  double h1 = (b1 - a1) / n1;
  double h2 = (b2 - a2) / n2;
  double integral = 0.0;

  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < n2; ++j) {
      double x0 = a1 + i * h1;
      double x1 = a1 + (i + 1) * h1;
      double y0 = a2 + j * h2;
      double y1 = a2 + (j + 1) * h2;

      double f00 = f(x0, y0);
      double f01 = f(x0, y1);
      double f10 = f(x1, y0);
      double f11 = f(x1, y1);

      integral += ((f00 + f01 + f10 + f11) / 4.0) * (x1 - x0) * (y1 - y0);
    }
  }

  return integral;
}

double trapezoidal_integral_omp(double a1, double b1, double a2, double b2, int n1, int n2,
                                const std::function<double(double, double)> &f) {
  double h1 = (b1 - a1) / n1;
  double h2 = (b2 - a2) / n2;
  double integral = 0.0;

#pragma omp parallel for reduction(+ : integral)
  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < n2; ++j) {
      double x0 = a1 + i * h1;
      double x1 = a1 + (i + 1) * h1;
      double y0 = a2 + j * h2;
      double y1 = a2 + (j + 1) * h2;

      double f00 = f(x0, y0);
      double f01 = f(x0, y1);
      double f10 = f(x1, y0);
      double f11 = f(x1, y1);

      integral += ((f00 + f01 + f10 + f11) / 4.0) * (x1 - x0) * (y1 - y0);
    }
  }

  return integral;
}

bool TestTaskSequentialMortinaIntegralTrapezoid::pre_processing() {
  internal_order_test();
  // Init value for input and output
  a1 = reinterpret_cast<double *>(taskData->inputs[0])[0];
  b1 = reinterpret_cast<double *>(taskData->inputs[0])[1];
  a2 = reinterpret_cast<double *>(taskData->inputs[0])[2];
  b2 = reinterpret_cast<double *>(taskData->inputs[0])[3];
  n1 = reinterpret_cast<int *>(taskData->inputs[1])[0];
  n2 = reinterpret_cast<int *>(taskData->inputs[1])[1];

  return true;
}

bool TestTaskSequentialMortinaIntegralTrapezoid::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 4 && taskData->inputs_count[1] == 2 && taskData->outputs_count[0] == 1;
}

bool TestTaskSequentialMortinaIntegralTrapezoid::run() {
  internal_order_test();
  res = trapezoidal_integral(a1, b1, a2, b2, n1, n2, fun);
  return true;
}

bool TestTaskSequentialMortinaIntegralTrapezoid::post_processing() {
  internal_order_test();
  reinterpret_cast<double *>(taskData->outputs[0])[0] = res;
  return true;
}

bool TestOMPTaskParallelMortinaIntegralTrapezoid::pre_processing() {
  internal_order_test();
  // Init vectors
  a1 = reinterpret_cast<double *>(taskData->inputs[0])[0];
  b1 = reinterpret_cast<double *>(taskData->inputs[0])[1];
  a2 = reinterpret_cast<double *>(taskData->inputs[0])[2];
  b2 = reinterpret_cast<double *>(taskData->inputs[0])[3];
  n1 = reinterpret_cast<int *>(taskData->inputs[1])[0];
  n2 = reinterpret_cast<int *>(taskData->inputs[1])[1];
  return true;
}

bool TestOMPTaskParallelMortinaIntegralTrapezoid::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 4 && taskData->inputs_count[1] == 2 && taskData->outputs_count[0] == 1;
}

bool TestOMPTaskParallelMortinaIntegralTrapezoid::run() {
  internal_order_test();
  res = trapezoidal_integral_omp(a1, b1, a2, b2, n1, n2, fun);
  return true;
}

bool TestOMPTaskParallelMortinaIntegralTrapezoid::post_processing() {
  internal_order_test();
  reinterpret_cast<double *>(taskData->outputs[0])[0] = res;
  return true;
}
}  // namespace Mortina_a_omp_integral_trapezoid