// Copyright 2024 Borisov Saveliy
#pragma once

#include <omp.h>

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace BorisovSaveliyOMP {
struct Point {
  int x, y;

  Point() = default;
  Point(const Point&) = default;
  Point(int x, int y);

  Point& operator=(const Point&) = default;

  bool operator==(const Point& other) const;

  bool operator!=(const Point& other) const;
};

class ConvexHull : public ppc::core::Task {
 public:
  explicit ConvexHull(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  uint8_t *input, *output;
  int height, width;
  std::vector<uint8_t> image;
  std::vector<Point> points;
  void convexHullImage();

  static std::vector<Point> convertToPoints(const std::vector<uint8_t>& _image, int _height, int _width);
  std::vector<int> convertToImageVector(const std::vector<Point>& _points, int _height, int _width);

  static int isLeft(const Point& p1, const Point& p2, const Point& point);
  static bool isCollinear(const Point& p1, const Point& p2, const Point& p3);
  static bool isOnSegment(const Point& p1, const Point& p2, const Point& point);
  static int windingNumber(const std::vector<Point>& polygon, const Point& point);
  static bool isInside(const std::vector<Point>& convexHull, const Point& point);
  static bool pointIsToTheRight(const Point& previous, const Point& current, const Point& potential);
};
}  // namespace BorisovSaveliyOMP
