// Copyright 2024 Kosarev Egor
#include "seq/kosarev_e_jarvis/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stack>
#include <vector>

namespace Kosarev_e_JarvisHull {

int orientation(const Point& p, const Point& q, const Point& r) {
  int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
  if (val == 0) return 0;    // collinear
  return (val > 0) ? 1 : 2;  // clock or counterclock
}

Point generateRandomPoint(int minX, int maxX, int minY, int maxY) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<int> disX(minX, maxX);
  std::uniform_int_distribution<int> disY(minY, maxY);
  return {disX(gen), disY(gen)};
}

std::vector<Point> generateRandomPoints(int numPoints, int minX, int maxX, int minY, int maxY) {
  std::vector<Point> points;
  points.reserve(numPoints);
  for (int i = 0; i < numPoints; ++i) {
    points.push_back(generateRandomPoint(minX, maxX, minY, maxY));
  }
  return points;
}

double distance(const Point& p1, const Point& p2) {
  return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

std::vector<Point> JarvisAlgo(const std::vector<Point>& arrPoints) {
  if (arrPoints.size() < 3) return arrPoints;

  auto startPoint = *std::min_element(arrPoints.begin(), arrPoints.end(), [](const Point& p, const Point& q) {
    return p.y < q.y || (p.y == q.y && p.x < q.x);
  });

  std::vector<Point> convexHull;
  convexHull.push_back(startPoint);

  Point prevPoint = startPoint;

  while (true) {
    Point nextPoint = arrPoints[0];
    for (const auto& point : arrPoints) {
      if (point == prevPoint) continue;
      int orient = orientation(prevPoint, nextPoint, point);
      if (orient == 2 || (orient == 0 && distance(prevPoint, point) > distance(prevPoint, nextPoint))) {
        nextPoint = point;
      }
    }

    if (nextPoint == startPoint) break;
    convexHull.push_back(nextPoint);
    prevPoint = nextPoint;
  }

  return convexHull;
}

bool TestTaskSequentialKosarevJarvisHull::pre_processing() {
  internal_order_test();
  // Init value for input and output
  points = std::vector<Point>(taskData->inputs_count[0]);

  auto* tmp_ptr_A = reinterpret_cast<Point*>(taskData->inputs[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    points[i] = tmp_ptr_A[i];
  }
  return true;
}

bool TestTaskSequentialKosarevJarvisHull::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] >= taskData->outputs_count[0];
}

bool TestTaskSequentialKosarevJarvisHull::run() {
  internal_order_test();
  pointsHull = JarvisAlgo(points);
  return true;
}

bool TestTaskSequentialKosarevJarvisHull::post_processing() {
  internal_order_test();
  std::copy(pointsHull.begin(), pointsHull.end(), reinterpret_cast<Point*>(taskData->outputs[0]));
  return true;
}

}  // namespace Kosarev_e_JarvisHull