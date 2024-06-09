// Copyright 2024 Shokurov Daniil
#include "tbb/shokurov_d_convex_hull/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <cmath>
#include <thread>
#include <utility>
#include <vector>

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
    auto* input_ = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);
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
    auto* outputs_ = reinterpret_cast<std::pair<double, double>*>(taskData->outputs[0]);
    for (size_t i = 0; i < si; ++i) {
      outputs_[i] = points[i];
    }
    taskData->outputs_count[0] = si;
    return true;
  } catch (...) {
    return false;
  }
}

struct PointFinder {
  const std::vector<std::pair<double, double>>& arr;
  const std::pair<double, double>& vec;
  const std::pair<double, double>& pk;
  const int k;
  int index;
  explicit PointFinder(const std::vector<std::pair<double, double>>& arr, const std::pair<double, double>& vec,
                       const std::pair<double, double>& pk, const int k)
      : arr(arr), vec(vec), pk(pk), k(k), index(k) {}
  PointFinder(const PointFinder& finder, tbb::split)
      : arr(finder.arr), vec(finder.vec), pk(finder.pk), k(finder.k), index(finder.k) {}
  void operator()(const oneapi::tbb::blocked_range<int>& r) {
    for (int i = r.begin(); i < r.end(); ++i) {
      if (!ConvexHullSequential::comp(arr[i], pk) &&
          ConvexHullSequential::my_less(ConvexHullSequential::sub(arr[i], pk),
                                        ConvexHullSequential::sub(arr[index], pk), vec)) {
        index = i;
      }
    }
  }
  void join(const PointFinder& finder) {
    if (!ConvexHullSequential::comp(arr[finder.index], pk) &&
        ConvexHullSequential::my_less(ConvexHullSequential::sub(arr[finder.index], pk),
                                      ConvexHullSequential::sub(arr[index], pk), vec)) {
      index = finder.index;
    }
  }
  int result() const { return index; }
};

size_t ConvexHullSequential::solve(std::vector<std::pair<double, double>>& p) {
  const int n = p.size();
  std::pair<double, double> p0 = p[index_lowest_right_point(p)];
  std::pair<double, double> vec = {1.0, 0.0};
  std::pair<double, double> pk = p0;
  int k = 0;
  do {
    PointFinder finder(p, vec, pk, k);
    /*int index = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<int>(k, n), k,
        [=](const oneapi::tbb::blocked_range<int>& r, int total) -> int {
          //std::cout << r.begin() << " " << r.end() << " " << total << "\n";
          for (int i = r.begin(); i < r.end(); ++i) {
            if (!comp(p[i], pk) && my_less(sub(p[i], pk), sub(p[total], pk), vec)) total = i;
          }
          return total;
        },
        [=](int a, int b) -> int {
          if (!comp(p[b], pk) && my_less(sub(p[a], pk), sub(p[b], pk), vec)) a = b;
          return a;
        });*/
    oneapi::tbb::parallel_reduce(oneapi::tbb::blocked_range<int>(k, n), finder);
    int index = finder.result();
    swap(p[index], p[k]);
    vec = sub(p[k], pk);
    pk = p[k];
    ++k;
  } while (!comp(pk, p0));
  return k;
}

bool ConvexHullSequential::comp(const std::pair<double, double>& a, const std::pair<double, double>& b) {
  return normal(sub(a, b)) < 1e-6;
}

bool ConvexHullSequential::my_less(const std::pair<double, double>& v1, const std::pair<double, double>& v2,
                                   const std::pair<double, double>& v) {
  const double cosa = cos(v1, v);
  const double cosb = cos(v2, v);
  bool flag = false;
  if (abs(cosa - cosb) > 1e-7)
    flag = cosa > cosb;
  else
    flag = normal(v1) > normal(v2);
  return flag;
}

std::pair<double, double> ConvexHullSequential::sub(const std::pair<double, double>& v1,
                                                    const std::pair<double, double>& v2) {
  return std::pair<double, double>(v1.first - v2.first, v1.second - v2.second);
}

size_t ConvexHullSequential::index_lowest_right_point(const std::vector<std::pair<double, double>>& v) {
  auto less = [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
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

double ConvexHullSequential::scalar_product(const std::pair<double, double>& v1, const std::pair<double, double>& v2) {
  return v1.first * v2.first + v1.second * v2.second;
}

double ConvexHullSequential::normal(const std::pair<double, double>& v) {
  return sqrt(v.first * v.first + v.second * v.second);
}

double ConvexHullSequential::cos(const std::pair<double, double>& v1, const std::pair<double, double>& v2) {
  const double n1 = normal(v1);
  const double n2 = normal(v2);
  return scalar_product(v1, v2) / (n1 * n2);
}
