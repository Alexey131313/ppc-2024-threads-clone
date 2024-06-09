// Copyright 2024 Shokurov Daniil
#include "omp/shokurov_d_convex_hull/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <thread>
#include <utility>
using namespace std::chrono_literals;

bool ConvexHullSequential::validation() {
  internal_order_test();
  try {
    if (taskData->inputs_count.size() != 1) throw 1;
    if (taskData->inputs_count[0] == 0) throw 2;
    if (taskData->outputs_count.size() != 1) throw 3;
    // if (taskData->outputs_count[0] != taskData->inputs_count[0]) throw 4;
    return true;
  } catch (...) {
    return false;
  }
}

bool ConvexHullSequential::pre_processing() {
  internal_order_test();
  try {
    // Init value for input and output
    auto* input_ = reinterpret_cast<pair<double, double>*>(taskData->inputs[0]);
    size_t n = taskData->inputs_count[0];
    points.assign(input_, input_ + n);
    return true;
  } catch (...) {
    return false;
  }
}

bool ConvexHullSequential::run() {
  internal_order_test();
  try {
    si = solve(points);
    return true;
  } catch (...) {
    return false;
  }
}

bool ConvexHullSequential::post_processing() {
  internal_order_test();
  try {
    auto* outputs_ = reinterpret_cast<pair<double, double>*>(taskData->outputs[0]);
    for (size_t i = 0; i < si; ++i) {
      outputs_[i] = points[i];
    }
    taskData->outputs_count[0] = si;
    return true;
  } catch (...) {
    return false;
  }
}

size_t ConvexHullSequential::solve(vector<pair<double, double>>& p) {
  const int n = p.size();
  pair<double, double> p0 = p[index_lowest_right_point(p)];
  pair<double, double> vec = {1.0, 0.0};
  pair<double, double> pk = p0;
  int k = 0;
  int common_index = 0;
#pragma omp parallel shared(k)
  {
    do {
#pragma omp single
      { common_index = k; }
      int index = k;
#pragma omp for schedule(static) nowait
      for (int i = k; i < n; ++i) {
        if (!comp(p[i], pk) && my_less(sub(p[i], pk), sub(p[index], pk), vec)) index = i;
      }
#pragma omp critical(common_index)
      {
        if (!comp(p[index], pk) && my_less(sub(p[index], pk), sub(p[common_index], pk), vec)) common_index = index;
      }
#pragma omp barrier
#pragma omp single
      {
        swap(p[common_index], p[k]);
        vec = sub(p[k], pk);
        pk = p[k];
        ++k;
      }
    } while (!comp(pk, p0));
  }
  return k;
}

bool ConvexHullSequential::comp(const pair<double, double>& a, const pair<double, double>& b) {
  return normal(sub(a, b)) < 1e-6;
}

bool ConvexHullSequential::my_less(const pair<double, double>& v1, const pair<double, double>& v2,
                                   const pair<double, double>& v) {
  const double cosa = cos(v1, v);
  const double cosb = cos(v2, v);
  bool flag = false;
  if (abs(cosa - cosb) > 1e-7)
    flag = cosa > cosb;
  else
    flag = normal(v1) > normal(v2);
  return flag;
}

pair<double, double> ConvexHullSequential::sub(const pair<double, double>& v1, const pair<double, double>& v2) {
  return pair<double, double>(v1.first - v2.first, v1.second - v2.second);
}

size_t ConvexHullSequential::index_lowest_right_point(const vector<pair<double, double>>& v) {
  auto less = [](const pair<double, double>& a, const pair<double, double>& b) {
    bool flag = false;
    if (a.second != b.second)
      flag = a.second < b.second;
    else
      flag = a.first > b.first;
    return flag;
  };
  size_t index = 0;
  if (!v.empty()) {
    for (size_t i = 1; i < v.size(); ++i) {
      if (less(v[i], v[index])) {
        index = i;
      }
    }
  } else {
    throw 11;
  }
  return index;
}

double ConvexHullSequential::scalar_product(const pair<double, double>& v1, const pair<double, double>& v2) {
  return v1.first * v2.first + v1.second * v2.second;
}

double ConvexHullSequential::normal(const pair<double, double>& v) {
  return sqrt(v.first * v.first + v.second * v.second);
}

double ConvexHullSequential::cos(const pair<double, double>& v1, const pair<double, double>& v2) {
  const double n1 = normal(v1);
  const double n2 = normal(v2);
  return scalar_product(v1, v2) / (n1 * n2);
}