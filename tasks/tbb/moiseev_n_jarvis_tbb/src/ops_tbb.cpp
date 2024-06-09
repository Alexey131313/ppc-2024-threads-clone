// Copyright 2024 Moiseev Nikita
#include "tbb/moiseev_n_jarvis_tbb/include/ops_tbb.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<Point> Jarvis_Moiseev(const std::vector<Point>& Points) {
  if (Points.size() < 3) return Points;

  Point p0 = Points[0];
  for (const auto& p : Points) {
    if (p.x < p0.x || (p.x == p0.x && p.y < p0.y)) p0 = p;
  }
  std::vector<Point> convexHull = {p0};
  Point prevPoint = p0;

  while (true) {
    Point nextPoint = Points[0];
    for (const auto& point : Points) {
      if (point == prevPoint) continue;

      double crossProduct =
          (point.y - prevPoint.y) * (nextPoint.x - prevPoint.x) - (point.x - prevPoint.x) * (nextPoint.y - prevPoint.y);

      if (crossProduct > 0 || (crossProduct == 0 && ((point.x - prevPoint.x) * (point.x - prevPoint.x) +
                                                     (point.y - prevPoint.y) * (point.y - prevPoint.y)) >
                                                        ((nextPoint.x - prevPoint.x) * (nextPoint.x - prevPoint.x) +
                                                         (nextPoint.y - prevPoint.y) * (nextPoint.y - prevPoint.y)))) {
        nextPoint = point;
      }
    }

    if (nextPoint == p0) break;
    convexHull.push_back(nextPoint);
    prevPoint = nextPoint;
  }

  return convexHull;
}

std::vector<Point> tbbJarvis_Moiseev(const std::vector<Point>& points) {
  size_t numberOfPoints = points.size();
  if (numberOfPoints < 3) return points;
  Point start = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(1, numberOfPoints), points[0],
      [&points](const tbb::blocked_range<size_t>& r, Point localStart) -> Point {
        auto begin = r.begin();
        auto end = r.end();
        for (auto i = begin; i != end; i++) {
          if (points[i] < localStart) localStart = points[i];
        }
        return localStart;
      },
      [](const Point& localStart, const Point& start) -> Point { return localStart < start ? localStart : start; });
  std::vector<Point> result;
  result.push_back(start);
  Point currentPoint = start;
  Point nextPoint;

  do {
    if (currentPoint == points[0]) {
      nextPoint = points[1];
    } else {
      nextPoint = points[0];
    }
    currentPoint = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, numberOfPoints), nextPoint,
        [&currentPoint, &points](const tbb::blocked_range<size_t>& r, Point localNext) -> Point {
          auto begin = r.begin();
          auto end = r.end();
          for (auto i = begin; i != end; i++) {
            int direction = (points[i].y - currentPoint.y) * (localNext.x - currentPoint.x) -
                            (points[i].x - currentPoint.x) * (localNext.y - currentPoint.y);
            if (direction > 0 ||
                (direction == 0 && ((points[i].x - currentPoint.x) * (points[i].x - currentPoint.x) +
                                    (points[i].y - currentPoint.y) * (points[i].y - currentPoint.y)) >
                                       ((localNext.x - currentPoint.x) * (localNext.x - currentPoint.x) +
                                        (localNext.y - currentPoint.y) * (localNext.y - currentPoint.y)))) {
              localNext = points[i];
            }
          }
          return localNext;
        },
        [&currentPoint](const Point& nextPoint, const Point& localNext) -> Point {
          int direction = (localNext.y - currentPoint.y) * (nextPoint.x - currentPoint.x) -
                          (localNext.x - currentPoint.x) * (nextPoint.y - currentPoint.y);
          if (direction > 0 ||
              (direction == 0 && ((localNext.x - currentPoint.x) * (localNext.x - currentPoint.x) +
                                  (localNext.y - currentPoint.y) * (localNext.y - currentPoint.y)) >
                                     ((nextPoint.x - currentPoint.x) * (nextPoint.x - currentPoint.x) +
                                      (nextPoint.y - currentPoint.y) * (nextPoint.y - currentPoint.y)))) {
            return localNext;
          }
          return nextPoint;
        });

    nextPoint = currentPoint;
    result.push_back(nextPoint);
  } while (currentPoint != start);

  result.pop_back();
  return result;
}
bool TestTaskTbbJarvisMoiseev::pre_processing() {
  internal_order_test();
  points.resize(taskData->inputs_count[0]);
  auto* tmp_ptr_A = reinterpret_cast<Point*>(taskData->inputs[0]);
  std::copy_n(tmp_ptr_A, taskData->inputs_count[0], points.begin());
  return true;
}

bool TestTaskTbbJarvisMoiseev::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] == 0) {
    return false;
  }
  std::vector<Point> expectedConvexHull = tbbJarvis_Moiseev(points);
  std::sort(expectedConvexHull.begin(), expectedConvexHull.end());
  return std::unique(expectedConvexHull.begin(), expectedConvexHull.end()) == expectedConvexHull.end();
}

bool TestTaskTbbJarvisMoiseev::run() {
  internal_order_test();
  convexHullPoints = tbbJarvis_Moiseev(points);
  return true;
}

bool TestTaskTbbJarvisMoiseev::post_processing() {
  internal_order_test();
  auto* output_ptr = reinterpret_cast<Point*>(taskData->outputs[0]);
  std::copy_n(convexHullPoints.begin(), convexHullPoints.size(), output_ptr);
  return true;
}

bool TestTaskSequentialJarvisMoiseev::pre_processing() {
  internal_order_test();
  points.resize(taskData->inputs_count[0]);
  auto* tmp_ptr_A = reinterpret_cast<Point*>(taskData->inputs[0]);
  std::copy_n(tmp_ptr_A, taskData->inputs_count[0], points.begin());
  return true;
}

bool TestTaskSequentialJarvisMoiseev::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] == 0) {
    return false;
  }

  std::sort(points.begin(), points.end());
  return std::unique(points.begin(), points.end()) == points.end();
}

bool TestTaskSequentialJarvisMoiseev::run() {
  internal_order_test();
  convexHullPoints = Jarvis_Moiseev(points);
  return true;
}

bool TestTaskSequentialJarvisMoiseev::post_processing() {
  internal_order_test();
  auto* output_ptr = reinterpret_cast<Point*>(taskData->outputs[0]);
  std::copy_n(convexHullPoints.begin(), convexHullPoints.size(), output_ptr);
  return true;
}
