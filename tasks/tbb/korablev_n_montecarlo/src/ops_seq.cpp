// Copyright 2024 Korablev Nikita
#include "tbb/korablev_n_montecarlo/include/ops_seq.hpp"

#include <omp.h>

#include <cmath>
#include <thread>

using namespace std::chrono_literals;

bool KorablevSequentialMonteCarlo::pre_processing() {
  internal_order_test();
  // Init value for input and output
  Int1[0] = reinterpret_cast<double*>(taskData->inputs[0])[0];
  Int1[1] = reinterpret_cast<double*>(taskData->inputs[0])[1];
  Int2[0] = reinterpret_cast<double*>(taskData->inputs[1])[0];
  Int2[1] = reinterpret_cast<double*>(taskData->inputs[1])[1];

  res = 0.0;
  return true;
}

bool KorablevSequentialMonteCarlo::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 2 && taskData->inputs_count[1] == 2 && taskData->outputs_count[0] == 1;
}

bool KorablevSequentialMonteCarlo::run() {
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

bool KorablevSequentialMonteCarlo::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}
