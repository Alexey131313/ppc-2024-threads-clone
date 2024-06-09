// Copyright 2024 Durandin Vladimir
#include "tbb/durandin_v_Jarvis/include/ops_tbb.hpp"

#include <thread>

static tbb::mutex my_mutex;

// Function to build convex hull using Jarvis's method
std::vector<Jarvis::Point2d> Jarvis::convexHullSeq(const std::vector<Jarvis::Point2d>& points) {
  int32_t n = points.size();
  if (n < 3) return std::vector<Jarvis::Point2d>{};  // Need at least 3 points

  // Function to determine orientation of three points
  auto orientation = [](Jarvis::Point2d p, Jarvis::Point2d q, Jarvis::Point2d r) {
    double val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val == 0.0) return 0;    // Collinear
    return (val > 0.0) ? 1 : 2;  // Clockwise or counterclockwise
  };

  // Find the leftmost lowest point
  int32_t leftmost = 0;
  for (int32_t i = 1; i < n; ++i) {
    if (points[i].x < points[leftmost].x)
      leftmost = i;
    else if (points[i].x == points[leftmost].x && points[i].y < points[leftmost].y)
      leftmost = i;
  }

  // Start traversal from the leftmost lowest point
  int32_t current = leftmost;
  int32_t next = 0;
  std::vector<Jarvis::Point2d> hull;

  do {
    hull.push_back(points[current]);
    next = (current + 1) % n;
    for (int32_t i = 0; i < n; ++i) {
      if (orientation(points[current], points[i], points[next]) == 2) next = i;
    }
    current = next;
  } while (current != leftmost);

  // Output vertices of the convex hull
  return hull;
}

std::vector<Jarvis::Point2d> Jarvis::convexHullTBB(const std::vector<Jarvis::Point2d>& points) {
  int32_t n = points.size();
  if (n < 3) return std::vector<Jarvis::Point2d>{};  // Need at least 3 points

  // Function to determine orientation of three points
  auto orientation = [](Jarvis::Point2d p, Jarvis::Point2d q, Jarvis::Point2d r) {
    double val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val == 0.0) return 0;    // Collinear
    return (val > 0.0) ? 1 : 2;  // Clockwise or counterclockwise
  };

  // Find the leftmost lowest point
  int32_t leftmost = 0;
  for (int32_t i = 1; i < n; ++i) {
    if (points[i].x < points[leftmost].x)
      leftmost = i;
    else if (points[i].x == points[leftmost].x && points[i].y < points[leftmost].y)
      leftmost = i;
  }

  // Start traversal from the leftmost lowest point
  int32_t current = leftmost;
  int32_t next = 0;
  std::vector<Jarvis::Point2d> hull;

  do {
    hull.push_back(points[current]);
    next = (current + 1) % n;
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, n), [&](const tbb::blocked_range<int32_t>& range) {
      for (int32_t i = range.begin(); i < range.end(); ++i) {
        if (orientation(points[current], points[i], points[next]) == 2) {
          tbb::mutex::scoped_lock lock(my_mutex);
          next = i;
        }
      }
    });
    current = next;

  } while (current != leftmost);

  // Output vertices of the convex hull
  return hull;
}

bool Jarvis::JarvisTestTaskSequential::pre_processing() {
  internal_order_test();
  try {
    points_count = taskData->inputs_count[0];
    points.resize(points_count);
    auto* ptr_points = reinterpret_cast<Jarvis::Point2d*>(taskData->inputs[0]);
    std::memcpy(points.data(), ptr_points, points_count * sizeof(Jarvis::Point2d));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return false;
  }
  return true;
}

bool Jarvis::JarvisTestTaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] >= 0) && (taskData->inputs[0] != nullptr);
}

bool Jarvis::JarvisTestTaskSequential::run() {
  internal_order_test();
  try {
    convex_hull = convexHullSeq(points);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return false;
  }
  return true;
}

bool Jarvis::JarvisTestTaskSequential::post_processing() {
  internal_order_test();
  try {
    auto* output_ptr = reinterpret_cast<Jarvis::Point2d*>(taskData->outputs[0]);
    std::memcpy(output_ptr, convex_hull.data(), convex_hull.size() * sizeof(Jarvis::Point2d));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return false;
  }
  return true;
}

bool Jarvis::JarvisTestTaskParallel::pre_processing() {
  internal_order_test();
  try {
    points_count = taskData->inputs_count[0];
    points.resize(points_count);
    auto* ptr_points = reinterpret_cast<Jarvis::Point2d*>(taskData->inputs[0]);
    std::memcpy(points.data(), ptr_points, points_count * sizeof(Jarvis::Point2d));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return false;
  }
  return true;
}

bool Jarvis::JarvisTestTaskParallel::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] >= 0) && (taskData->inputs[0] != nullptr);
}

bool Jarvis::JarvisTestTaskParallel::run() {
  internal_order_test();
  try {
    convex_hull = convexHullTBB(points);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return false;
  }
  return true;
}

bool Jarvis::JarvisTestTaskParallel::post_processing() {
  internal_order_test();
  try {
    auto* output_ptr = reinterpret_cast<Jarvis::Point2d*>(taskData->outputs[0]);
    std::memcpy(output_ptr, convex_hull.data(), convex_hull.size() * sizeof(Jarvis::Point2d));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return false;
  }
  return true;
}
