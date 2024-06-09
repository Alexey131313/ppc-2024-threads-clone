// Copyright 2024 Ivlev Alexander
#include "omp/ivlev_a_convex_hull/include/ops_omp.hpp"

#include <omp.h>

#include <thread>

using namespace std::chrono_literals;
using namespace ivlev_a_omp;

bool ConvexHullOMPTaskParallel::pre_processing() {
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

bool ConvexHullOMPTaskParallel::validation() {
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

bool ConvexHullOMPTaskParallel::run() {
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

bool ConvexHullOMPTaskParallel::post_processing() {
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

inline size_t ivlev_a_omp::rotation(const std::pair<ptrdiff_t, ptrdiff_t>& a, const std::pair<ptrdiff_t, ptrdiff_t>& b,
                                    const std::pair<ptrdiff_t, ptrdiff_t>& c) {
  int tmp = (b.first - a.first) * (c.second - b.second) - (b.second - a.second) * (c.first - b.first);
  if (tmp == 0) return 0;
  return ((tmp > 0) ? 1 : 2);
}

std::vector<std::pair<size_t, size_t>> ConvexHullOMPTaskParallel::Convex_Hull_tmp(
    size_t p, size_t last, size_t n, const std::vector<std::pair<size_t, size_t>>& component_) {
  std::vector<std::pair<size_t, size_t>> res = {};
  size_t q = 0;

  do {
    q = (p + 1) % n;

    for (size_t i = 0; i < n; i++) {
      size_t tmp_r = rotation(component_[p], component_[i], component_[q]);
      if (tmp_r == 2 || (tmp_r == 0 && i > p && i != q)) {
        q = i;
      }
    }

    res.push_back(component_[q]);
    p = q;
  } while (p != last);

  return res;
}

std::vector<std::pair<size_t, size_t>> ConvexHullOMPTaskParallel::Convex_Hull(
    const std::vector<std::pair<size_t, size_t>>& component_) {
  size_t n = component_.size();
  if (n < 3) return component_;

  size_t left = 0;
  size_t top = 0;
  size_t rigth = 0;
  size_t down = 0;

  if (n < 8) {
    for (int i = 1; i < (int)n; i++) {
      if (component_[i].second < component_[left].second) left = i;
    }
    return ConvexHullOMPTaskParallel::Convex_Hull_tmp(left, left, n, component_);
  }

  std::vector<std::pair<size_t, size_t>> res = {};

#pragma omp parallel num_threads(4) shared(left, top, rigth, down, res) firstprivate(n)
  {
    size_t thr_left = 0;
    size_t thr_top = 0;
    size_t thr_rigth = 0;
    size_t thr_down = 0;
#pragma omp for nowait
    for (int i = 1; i < (int)n; i++) {
      if (component_[i].second <= component_[thr_left].second) thr_left = i;
      if (component_[i].first < component_[thr_top].first) thr_top = i;
      if (component_[i].second > component_[thr_rigth].second) thr_rigth = i;
      if (component_[i].first >= component_[thr_down].first) thr_down = i;
    }

#pragma omp critical
    {
      if (component_[thr_left].second < component_[left].second ||
          (component_[thr_left].second == component_[left].second && thr_left > left))
        left = thr_left;
      if (component_[thr_top].first < component_[top].first ||
          (component_[thr_top].first == component_[top].first && thr_top < top))
        top = thr_top;
      if (component_[thr_rigth].second > component_[rigth].second ||
          (component_[thr_rigth].second == component_[rigth].second && thr_rigth < rigth))
        rigth = thr_rigth;
      if (component_[thr_down].first > component_[down].first ||
          (component_[thr_down].first == component_[down].first && thr_down > down))
        down = thr_down;
    }

#pragma omp barrier

#pragma omp sections
    {
#pragma omp section
      {
        std::vector<std::pair<size_t, size_t>> thr_res =
            ConvexHullOMPTaskParallel::Convex_Hull_tmp(left, top, n, component_);
#pragma omp critical
        { res.insert(res.end(), thr_res.begin(), thr_res.end()); }
      }
#pragma omp section
      {
        std::vector<std::pair<size_t, size_t>> thr_res =
            ConvexHullOMPTaskParallel::Convex_Hull_tmp(top, rigth, n, component_);
#pragma omp critical
        { res.insert(res.end(), thr_res.begin(), thr_res.end()); }
      }
#pragma omp section
      {
        std::vector<std::pair<size_t, size_t>> thr_res =
            ConvexHullOMPTaskParallel::Convex_Hull_tmp(rigth, down, n, component_);
#pragma omp critical
        { res.insert(res.end(), thr_res.begin(), thr_res.end()); }
      }
#pragma omp section
      {
        std::vector<std::pair<size_t, size_t>> thr_res =
            ConvexHullOMPTaskParallel::Convex_Hull_tmp(down, left, n, component_);
#pragma omp critical
        { res.insert(res.end(), thr_res.begin(), thr_res.end()); }
      }
    }
  }

  return res;
}
