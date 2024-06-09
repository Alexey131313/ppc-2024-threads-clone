// Copyright 2024 Mamaeva Olga
#include "seq/mamaeva_o_jarvis/include/ops_seq.hpp"

#define M_PI_VALUE 3.14159265358979323846

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <utility>
#include <vector>

void jarvis::prepare_points(jarvis::r* points, int n, jarvis::r* hull, int h, double r, unsigned int seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution gen(-r, r);
  int i = 0;
  double phi = 2 * M_PI_VALUE / h;
  for (i = 0; i < n; i++) {
    points[i] = {gen(g), gen(g)};
  }
  int k = n / (h + 1);
  for (i = 0; i < h; i++) {
    hull[i] = points[k * (i + 1)] = {cos(phi * i) * 2 * r, sin(phi * i) * 2 * r};
  }
}

bool JarvisHullSequential::pre_processing() {
  internal_order_test();
  auto* p = reinterpret_cast<jarvis::r*>(taskData->inputs[0]);
  points = std::vector<jarvis::r>(p, p + taskData->inputs_count[0]);
  hull = reinterpret_cast<jarvis::r*>(taskData->outputs[0]);
  hull_sz = taskData->outputs_count[0];
  return true;
}

bool JarvisHullSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs[0] != nullptr) && (taskData->inputs_count[0] != 0u) && (taskData->outputs[0] != nullptr) &&
         (taskData->outputs_count[0] != 0u);
}

void Jarvis(const std::vector<jarvis::r>& points, jarvis::r* hull) {
  int n = points.size();
  int left = 0;
  jarvis::r p_point = points[left];
  for (int i = 1; i < n; i++)
    if (points[i] < p_point) {
      p_point = points[left = i];
    }

  int p = left;
  int q;
  int h_i = 0;
  while (true) {
    q = (p + 1) % n;
    for (int i = 0; i < n; i++) {
      if (((p_point - points[i]) ^ (points[q] - points[i])) < -1e-7) q = i;
    }
    hull[h_i++] = points[q];
    p_point = points[p = q];
    if (p == left) break;
  }
}

bool JarvisHullSequential::run() {
  internal_order_test();
  Jarvis(points, hull);
  return true;
}

bool JarvisHullSequential::post_processing() {
  internal_order_test();
  return true;
}
