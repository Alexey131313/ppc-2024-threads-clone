// Copyright 2024 Pozdnyakov Vasya
#include "tbb/pozdnyakov_v_rect_integral/include/ops_tbb.hpp"

#include <oneapi/tbb.h>

#include <cstdlib>

double pozdnyakov_tbb::pozdnyakov_flin(double x, double y) { return x - y; }
double pozdnyakov_tbb::pozdnyakov_fxy(double x, double y) { return x * y; }
double pozdnyakov_tbb::pozdnyakov_fysinx(double x, double y) { return y * std::sin(x); }
double pozdnyakov_tbb::pozdnyakov_fxexpy(double x, double y) { return x * std::exp(y); }

bool pozdnyakov_tbb::PozdnyakovTaskTBB::pre_processing() {
  internal_order_test();
  try {
    auto* tmp = reinterpret_cast<double*>(taskData->inputs[0]);
    x1 = tmp[0], x2 = tmp[1], y1 = tmp[2], y2 = tmp[3];
    res = 0.0;
    f = reinterpret_cast<Func>(taskData->inputs[1]);
    n = reinterpret_cast<int*>(taskData->inputs[2])[0];
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

bool pozdnyakov_tbb::PozdnyakovTaskTBB::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 4 && taskData->outputs_count[0] == 1;
}

bool pozdnyakov_tbb::PozdnyakovTaskTBB::run() {
  internal_order_test();
  try {
    double x_i = std::abs(x2 - x1) / n;
    double y_i = std::abs(y2 - y1) / n;
    res = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, n), 0.0,
        [&](const tbb::blocked_range<int>& r, double local_res) {
          for (int i = r.begin(); i != r.end(); ++i) {
            for (int j = 0; j < n; ++j) {
              local_res += f(x1 + i * x_i, y1 + j * y_i);
            }
          }
          return local_res;
        },
        std::plus<>());
    res *= x_i * y_i;
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

bool pozdnyakov_tbb::PozdnyakovTaskTBB::post_processing() {
  internal_order_test();
  try {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

bool pozdnyakov_tbb::PozdnyakovTaskSequential::pre_processing() {
  internal_order_test();
  try {
    auto* tmp = reinterpret_cast<double*>(taskData->inputs[0]);
    x1 = tmp[0], x2 = tmp[1], y1 = tmp[2], y2 = tmp[3];
    res = 0.0;
    f = reinterpret_cast<Func>(taskData->inputs[1]);
    n = reinterpret_cast<int*>(taskData->inputs[2])[0];
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

bool pozdnyakov_tbb::PozdnyakovTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 4 && taskData->outputs_count[0] == 1;
}

bool pozdnyakov_tbb::PozdnyakovTaskSequential::run() {
  internal_order_test();
  try {
    double x_i = std::abs(x2 - x1) / n;
    double y_i = std::abs(y2 - y1) / n;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        res += f(x1 + i * x_i, y1 + j * y_i);
      }
    }
    res *= x_i * y_i;
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}

bool pozdnyakov_tbb::PozdnyakovTaskSequential::post_processing() {
  internal_order_test();
  try {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return false;
  }
  return true;
}
