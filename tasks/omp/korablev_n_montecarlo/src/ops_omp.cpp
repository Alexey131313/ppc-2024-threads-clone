// Copyright 2024 Korablev Nikita
#include "omp/korablev_n_montecarlo/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <thread>

using namespace std::chrono_literals;

bool KorablevOMPMonteCarlo::pre_processing() {
  internal_order_test();
  // Init value for input and output
  Int1[0] = reinterpret_cast<double*>(taskData->inputs[0])[0];
  Int1[1] = reinterpret_cast<double*>(taskData->inputs[0])[1];
  Int2[0] = reinterpret_cast<double*>(taskData->inputs[1])[0];
  Int2[1] = reinterpret_cast<double*>(taskData->inputs[1])[1];
  function = reinterpret_cast<korablev_func>(taskData->inputs[2]);

  N = 1000;
  res = 0.0;
  return true;
}

bool KorablevOMPMonteCarlo::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 2 && taskData->inputs_count[1] == 2 && taskData->outputs_count[0] == 1;
}

bool KorablevOMPMonteCarlo::run() {
  internal_order_test();

  try {
    double h1 = (Int1[1] - Int1[0]) / N;
    double h2 = (Int2[1] - Int2[0]) / N;

    double tmp_res = .0;
#pragma omp parallel for reduction(+ : tmp_res)
    for (int j = 0; j < N; ++j) {
      double y = Int2[0] + h2 * (double)j;
      for (int i = 0; i < N; ++i) {
        tmp_res += function(Int1[0] + h1 * (double)i, y);
      }
    }
    res = tmp_res * h1 * h2;
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return false;
  }

  return true;
}

bool KorablevOMPMonteCarlo::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}
