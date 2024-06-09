// Copyright 2024 Kriseev Mikhail
#include "seq/kriseev_m_convex_hull_graham/include/kriseev_m_convex_hull_seq.hpp"

#include <algorithm>
#include <limits>

double angle1(const KriseevMTaskSeq::Point &origin, const KriseevMTaskSeq::Point &point) {
  double dx = point.first - origin.first;
  double dy = point.second - origin.second;
  if (dx == 0.0 && dy == 0.0) {
    // Return this incorrect value to ensure the origin is always first in sorted array
    return -1.0;
  }
  // Calculating angle using algorithm from https://stackoverflow.com/a/14675998/14116050
  return (dy >= 0) ? (dx >= 0 ? dy / (dx + dy) : 1 - dx / (-dx + dy))
                   : (dx < 0 ? 2 - dy / (-dx - dy) : 3 + dx / (dx - dy));
}
bool compareForSort1(const KriseevMTaskSeq::Point &origin, const KriseevMTaskSeq::Point &a,
                     const KriseevMTaskSeq::Point &b) {
  double angleA = angle1(origin, a);
  double angleB = angle1(origin, b);
  if (angleA < angleB) {
    return true;
  }
  if (angleA > angleB) {
    return false;
  }
  double dxA = a.first - origin.first;
  double dyA = a.second - origin.second;
  double dxB = b.first - origin.first;
  double dyB = b.second - origin.second;
  // The further point must be first
  // so the others will be ignored
  return dxA * dxA + dyA * dyA > dxB * dxB + dyB * dyB;
}

bool checkOrientation(const KriseevMTaskSeq::Point &origin, const KriseevMTaskSeq::Point &a,
                      const KriseevMTaskSeq::Point &b) {
  double dxA = a.first - origin.first;
  double dyA = a.second - origin.second;
  double dxB = b.first - origin.first;
  double dyB = b.second - origin.second;
  // Using cross product to check orientation
  return dxA * dyB - dyA * dxB < 0;
}

bool KriseevMTaskSeq::ConvexHullTask::pre_processing() {
  internal_order_test();
  auto *pointsX = reinterpret_cast<double *>(taskData->inputs.at(0));
  auto *pointsY = reinterpret_cast<double *>(taskData->inputs.at(1));
  points = std::vector<std::pair<double, double>>();
  points.reserve(taskData->inputs_count[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; ++i) {
    points.emplace_back(pointsX[i], pointsY[i]);
  }

  return true;
}

bool KriseevMTaskSeq::ConvexHullTask::validation() {
  internal_order_test();
  if (taskData->inputs_count.at(0) != taskData->inputs_count.at(1)) {
    return false;
  }
  if (taskData->inputs_count[0] < 3) {
    return false;
  }
  return true;
}

bool KriseevMTaskSeq::ConvexHullTask::run() {
  internal_order_test();
  auto originIt =
      std::min_element(points.begin(), points.end(), [](auto &a, auto &b) -> bool { return a.second < b.second; });
  auto origin = *originIt;

  for (size_t phase = 0; phase < points.size(); phase++) {
    if ((phase & 1) == 0) {
      for (size_t i = 1; i < points.size(); i += 2) {
        if (compareForSort1(origin, points[i], points[i - 1])) {
          std::iter_swap(points.begin() + i, points.begin() + i - 1);
        }
      }
    } else {
      for (size_t i = 1; i < points.size() - 1; i += 2) {
        if (compareForSort1(origin, points[i + 1], points[i])) {
          std::iter_swap(points.begin() + i, points.begin() + i + 1);
        }
      }
    }
  }

  std::vector<Point> hull;
  hull.reserve(points.size());
  hull.push_back(points[0]);
  hull.push_back(points[1]);
  hull.push_back(points[2]);

  for (uint32_t i = 3; i < points.size(); ++i) {
    while (hull.size() > 1 && !checkOrientation(hull.back(), *(hull.end() - 2), points[i])) {
      hull.pop_back();
    }
    hull.push_back(points[i]);
  }
  this->finalHull = hull;
  return true;
}

bool KriseevMTaskSeq::ConvexHullTask::post_processing() {
  internal_order_test();
  auto *resultsX = reinterpret_cast<double *>(taskData->outputs[0]);
  auto *resultsY = reinterpret_cast<double *>(taskData->outputs[1]);
  for (size_t i = 0; i < finalHull.size(); ++i) {
    resultsX[i] = finalHull[i].first;
    resultsY[i] = finalHull[i].second;
  }
  taskData->outputs_count[0] = finalHull.size();
  taskData->outputs_count[1] = finalHull.size();
  return true;
}
