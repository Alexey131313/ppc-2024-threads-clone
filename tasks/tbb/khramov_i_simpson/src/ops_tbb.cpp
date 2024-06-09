// Copyright 2024 Khramov Ivan
#include "tbb/khramov_i_simpson/include/ops_tbb.hpp"

#include <oneapi/tbb.h>

#include <thread>

using namespace std::chrono_literals;

double khramov_tbb::simpson_formula(function func, double Xj0, double Xj1, double Xi) {
  return func(Xj0, Xi) + 4 * func((Xj0 + Xj1) / 2, Xi) + func(Xj1, Xi);
}

double khramov_tbb::simpson_seq(function func, int a1, int a2, int b1, int b2, int numSteps) {
  double res{};
  double h1 = static_cast<double>(b1 - a1) / numSteps;
  double h2 = static_cast<double>(b2 - a2) / numSteps;

  for (int i = 0; i < numSteps; i++) {
    double Xi0 = a2 + i * h2;
    double Xi1 = a2 + (i + 1) * h2;
    for (int j = 0; j < numSteps; j++) {
      double Xj0 = a1 + j * h1;
      double Xj1 = a1 + (j + 1) * h1;

      res += (h1 * h2 / 36) *
             (simpson_formula(func, Xj0, Xj1, Xi0) + 4 * simpson_formula(func, Xj0, Xj1, (Xi0 + Xi1) / 2) +
              simpson_formula(func, Xj0, Xj1, Xi1));
    }
  }

  return res;
}

double khramov_tbb::simpson_parallel(function func, int a1, int a2, int b1, int b2, int numSteps) {
  double res{};
  double h1 = static_cast<double>(b1 - a1) / numSteps;
  double h2 = static_cast<double>(b2 - a2) / numSteps;

  res = oneapi::tbb::parallel_reduce(
      oneapi::tbb::blocked_range<int>(0, numSteps), 0.0,
      [&](const tbb::blocked_range<int>& range, double temp) {
        for (int i = range.begin(); i != range.end(); i++) {
          double Xi0 = a2 + i * h2;
          double Xi1 = a2 + (i + 1) * h2;
          for (int j = 0; j < numSteps; j++) {
            double Xj0 = a1 + j * h1;
            double Xj1 = a1 + (j + 1) * h1;

            temp += (h1 * h2 / 36) *
                    (simpson_formula(func, Xj0, Xj1, Xi0) + 4 * simpson_formula(func, Xj0, Xj1, (Xi0 + Xi1) / 2) +
                     simpson_formula(func, Xj0, Xj1, Xi1));
          }
        }
        return temp;
      },
      std::plus<>());

  return res;
}

double khramov_tbb::linear_f(double x0, double x1) { return 3 * x0 - 2 * x1; }
double khramov_tbb::square_f(double x0, double x1) { return x0 * x0 + 2 * x1 * x1; }
double khramov_tbb::sin_f(double x0, double x1) { return sin(x0 + x1); }
double khramov_tbb::mult_f(double x0, double x1) { return x0 * x1; }
double khramov_tbb::exp_f(double x0, double x1) { return exp(x0 * x1); }

bool khramov_tbb::TestSimpsonTbbsequential::pre_processing() {
  internal_order_test();

  auto* data = reinterpret_cast<int*>(taskData->inputs[0]);
  a1 = data[0];
  a2 = data[1];
  b1 = data[2];
  b2 = data[3];
  num_steps = data[4];

  result = 0.0;
  return true;
}

bool khramov_tbb::TestSimpsonTbbsequential::validation() {
  internal_order_test();

  return taskData->inputs_count[0] == 5 && taskData->outputs_count[0] == 1;
}

bool khramov_tbb::TestSimpsonTbbsequential::run() {
  internal_order_test();

  result = simpson_seq(func, a1, a2, b1, b2, num_steps);
  return true;
}

bool khramov_tbb::TestSimpsonTbbsequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  return true;
}

bool khramov_tbb::TestSimpsonTbbParallel::pre_processing() {
  internal_order_test();

  auto* data = reinterpret_cast<int*>(taskData->inputs[0]);
  a1 = data[0];
  a2 = data[1];
  b1 = data[2];
  b2 = data[3];
  num_steps = data[4];

  result = 0.0;
  return true;
}

bool khramov_tbb::TestSimpsonTbbParallel::validation() {
  internal_order_test();

  return taskData->inputs_count[0] == 5 && taskData->outputs_count[0] == 1;
}

bool khramov_tbb::TestSimpsonTbbParallel::run() {
  internal_order_test();

  result = simpson_parallel(func, a1, a2, b1, b2, num_steps);
  return true;
}

bool khramov_tbb::TestSimpsonTbbParallel::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  return true;
}
