// Copyright 2024 Ivlev Alexander
#include "seq/ivlev_a_convex_hull/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;
using namespace ivlev_a_seq;

bool ConvexHullSequential::pre_processing() {
  internal_order_test();
  try {
    size_t n = taskData->inputs_count[0];
    components.resize(n);
    for (size_t i = 0; i < n; i++) {
      auto* input_ = reinterpret_cast<std::pair<size_t, size_t>*>(taskData->inputs[i]);
      size_t tmp_size = taskData->inputs_count[i + 1];
      components[i].assign(input_, input_ + tmp_size);
      size_t m_w = 0;

      for (size_t j = 0; j < tmp_size; j++) {
        if (components[i][j].second > m_w) {
          m_w = components[i][j].second;
        }
      }

      sizes.emplace_back(components[i].back().first + 1, m_w + 1);
    }
    results.resize(taskData->inputs_count[0]);
  } catch (...) {
    std::cout << "pre\n";
    return false;
  }
  return true;
}

bool ConvexHullSequential::validation() {
  internal_order_test();
  try {
    if (taskData->inputs_count.size() <= 1) return false;
    if (taskData->outputs_count.empty()) return false;
    if (taskData->inputs_count[0] < 1) return false;
    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      if (taskData->inputs[i] == nullptr) return false;
      if (taskData->inputs_count[i] < 1) return false;
    }
    if (taskData->outputs[0] == nullptr) return false;

  } catch (...) {
    std::cout << "val\n";
    return false;
  }

  return true;
}

bool ConvexHullSequential::run() {
  internal_order_test();
  try {
    for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
      results[i] = Convex_Hull(components[i]);
    }
  } catch (...) {
    std::cout << "run\n";
    return false;
  }
  return true;
}

bool ConvexHullSequential::post_processing() {
  internal_order_test();
  try {
    size_t n = taskData->inputs_count[0];
    auto* outputs_ = reinterpret_cast<std::vector<std::pair<size_t, size_t>>*>(taskData->outputs[0]);
    for (size_t i = 0; i < n; i++) {
      std::sort(results[i].begin(), results[i].end());
      outputs_[i].clear();
      for (size_t j = 0; j < results[i].size(); j++) {
        outputs_[i].push_back(results[i][j]);
      }
    }
  } catch (...) {
    std::cout << "post\n";
    return false;
  }
  return true;
}

size_t ivlev_a_seq::rotation(const std::pair<int, int>& a, const std::pair<int, int>& b, const std::pair<int, int>& c) {
  int tmp = (b.first - a.first) * (c.second - b.second) - (b.second - a.second) * (c.first - b.first);
  if (tmp == 0) return 0;
  return ((tmp > 0) ? 1 : 2);
}

std::vector<std::pair<size_t, size_t>> ConvexHullSequential::Convex_Hull(
    const std::vector<std::pair<size_t, size_t>>& component_) {
  size_t n = component_.size();
  if (n < 3) return component_;

  size_t left = 0;
  for (size_t i = 1; i < n; i++)
    if (component_[i].second < component_[left].second) left = i;

  std::vector<std::pair<size_t, size_t>> res = {};
  size_t p = left;
  do {
    size_t q = (p + 1) % n;
    for (size_t i = 0; i < n; i++) {
      if (rotation(component_[p], component_[i], component_[q]) == 2 ||
          (rotation(component_[p], component_[i], component_[q]) == 0 && i > p && i != q)) {
        q = i;
      }
    }
    res.push_back(component_[q]);
    p = q;
  } while (p != left);

  return res;
}
