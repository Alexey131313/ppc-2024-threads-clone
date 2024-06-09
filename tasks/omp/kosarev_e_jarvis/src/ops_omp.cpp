// Copyright 2024 Kosarev Egor
#include "omp/kosarev_e_jarvis/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace Kosarev_e_OMP_KosarevJarvisHull {

int orientation(const Point& p, const Point& q, const Point& r) {
  int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
  if (val == 0) return 0;    // collinear
  return (val > 0) ? 1 : 2;  // clock or counterclock wise
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

std::vector<Point> JarvisAlgo_omp(const std::vector<Point>& arrPoints, int threadsNom) {
  std::vector<Point> res;
  omp_set_num_threads(threadsNom);

#pragma omp parallel
  {
    int threadNom = omp_get_thread_num();
    int localSize;
    int delta = arrPoints.size() / threadsNom;
    int remains = arrPoints.size() % threadsNom;
    std::vector<Point> localVector;

    if (threadNom == 0) {
      localSize = remains + delta;
      localVector = std::vector<Point>(arrPoints.begin(), arrPoints.begin() + localSize);
    } else {
      localSize = arrPoints.size() / threadsNom;
      localVector = std::vector<Point>(arrPoints.begin() + (localSize * threadNom) + remains,
                                       arrPoints.begin() + (localSize * (threadNom + 1)) + remains);
    }

    std::vector<Point> localRes = JarvisAlgo(localVector);

#pragma omp critical
    { res.insert(res.end(), localRes.begin(), localRes.end()); }
  }

  return JarvisAlgo(res);
}

bool TestTaskSequentialKosarevJarvisHull::pre_processing() {
  internal_order_test();
  // Init value for input and output
  points = std::vector<Point>(taskData->inputs_count[0]);

  auto* tmp_ptr_A = reinterpret_cast<Point*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; ++i) {
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

bool TestOMPTaskParallelKosarevJarvisHull::pre_processing() {
  internal_order_test();
  // Init value for input and output
  points = std::vector<Point>(taskData->inputs_count[0]);

  auto* tmp_ptr_A = reinterpret_cast<Point*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; ++i) {
    points[i] = tmp_ptr_A[i];
  }
  return true;
}

bool TestOMPTaskParallelKosarevJarvisHull::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] >= taskData->outputs_count[0];
}

bool TestOMPTaskParallelKosarevJarvisHull::run() {
  internal_order_test();
  pointsHull = JarvisAlgo_omp(points, 5);
  return true;
}
bool TestOMPTaskParallelKosarevJarvisHull::post_processing() {
  internal_order_test();
  std::copy(pointsHull.begin(), pointsHull.end(), reinterpret_cast<Point*>(taskData->outputs[0]));
  return true;
}

}  // namespace Kosarev_e_OMP_KosarevJarvisHull