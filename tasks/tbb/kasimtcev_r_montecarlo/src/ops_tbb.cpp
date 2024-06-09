// Copyright 2024 Kasimtcev Roman
#include "tbb/kasimtcev_r_montecarlo/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <thread>

#include "tbb/kasimtcev_r_montecarlo/include/my_funcs.hpp"

using namespace std::chrono_literals;

bool KasimtcevMonteCarloTBB::pre_processing() {
  internal_order_test();
  Int1[0] = reinterpret_cast<double*>(taskData->inputs[0])[0];
  Int1[1] = reinterpret_cast<double*>(taskData->inputs[0])[1];
  Int2[0] = reinterpret_cast<double*>(taskData->inputs[1])[0];
  Int2[1] = reinterpret_cast<double*>(taskData->inputs[1])[1];

  res = 0.0;
  return true;
}

bool KasimtcevMonteCarloTBB::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 2 && taskData->inputs_count[1] == 2 && taskData->outputs_count[0] == 1;
}

bool KasimtcevMonteCarloTBB::run() {
  internal_order_test();

  double h1 = (Int1[1] - Int1[0]) / N;
  double h2 = (Int2[1] - Int2[0]) / N;

  res = oneapi::tbb::parallel_reduce(
      oneapi::tbb::blocked_range<int>(0, N), 0.0,
      [&](const tbb::blocked_range<int>& r, double total) {
        for (int j = r.begin(); j < r.end(); ++j) {
          double y = Int2[0] + h2 * j;
          for (int i = 0; i < N; ++i) {
            total += function(Int1[0] + h1 * i, y);
          }
        }
        return total;
      },
      std::plus<>());
  res *= h1 * h2;

  return true;
}

bool KasimtcevMonteCarloTBB::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}
