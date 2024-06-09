// Copyright 2024 Durandin Vladimir
#include <gtest/gtest.h>

#include "seq/durandin_v_Jarvis/include/ops_seq.hpp"

TEST(VladimirD_Sequential, No_Points) {
  // Create data
  std::vector<Jarvis::Point2d> in{};
  std::vector<Jarvis::Point2d> expectedHull{};
  std::vector<Jarvis::Point2d> out(expectedHull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(in.size()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(out.size()));

  // Create Task
  Jarvis::JarvisTestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(VladimirD_Sequential, Collinear_Points) {
  // Create data
  std::vector<Jarvis::Point2d> in{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};
  std::vector<Jarvis::Point2d> expectedHull{{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}}};
  std::vector<Jarvis::Point2d> out(expectedHull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(in.size()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(out.size()));

  // Create Task
  Jarvis::JarvisTestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_EQ(expectedHull[i].x, out[i].x);
    EXPECT_EQ(expectedHull[i].y, out[i].y);
  }
}

TEST(VladimirD_Sequential, Convex_Hull_5_Different_Points) {
  // Create data
  std::vector<Jarvis::Point2d> in = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {1, 1}};
  std::vector<Jarvis::Point2d> expectedHull = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
  std::vector<Jarvis::Point2d> out(expectedHull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(in.size()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(out.size()));

  // Create Task
  Jarvis::JarvisTestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_EQ(expectedHull[i].x, out[i].x);
    EXPECT_EQ(expectedHull[i].y, out[i].y);
  }
}

TEST(VladimirD_Sequential, Convex_Hull_Circle_10_points) {
  auto generatePointsInCircle = [](const Jarvis::Point2d& center, double radius, uint32_t numPoints) {
    std::vector<Jarvis::Point2d> points;
    double angleIncrement = 2 * Const::MY_PI / numPoints;
    for (uint32_t i = 0; i < numPoints; ++i) {
      double angle = i * angleIncrement;
      double x = center.x + radius * std::cos(angle);
      double y = center.y + radius * std::sin(angle);
      points.push_back({x, y});
    }
    return points;
  };

  Jarvis::Point2d center{0, 0};
  double radius = 5.0;
  uint32_t numPoints = 10;  // Number of points for the circle

  std::vector<Jarvis::Point2d> points = generatePointsInCircle(center, radius, numPoints);
  // Expected result: the circle should be a convex hull
  std::vector<Jarvis::Point2d> expectedHull = points;  // Since the circle is already a convex hull

  std::vector<Jarvis::Point2d> out(points.size());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(points.size()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(out.size()));

  // Create Task
  Jarvis::JarvisTestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);

  // Such a strange check because my algorithm outputs points in a different order
  for (uint32_t i = 0; i < out.size(); ++i) {
    if (i < 5) {
      EXPECT_EQ(expectedHull[i].x, out[i + 5].x);
      EXPECT_EQ(expectedHull[i].y, out[i + 5].y);
    } else {
      EXPECT_EQ(expectedHull[i].x, out[i - 5].x);
      EXPECT_EQ(expectedHull[i].y, out[i - 5].y);
    }
  }
}

TEST(VladimirD_Sequential, Convex_Hull_Circle_100_points) {
  auto generatePointsInCircle = [](const Jarvis::Point2d& center, double radius, uint32_t numPoints) {
    std::vector<Jarvis::Point2d> points;
    double angleIncrement = 2 * Const::MY_PI / numPoints;
    for (uint32_t i = 0; i < numPoints; ++i) {
      double angle = i * angleIncrement;
      double x = center.x + radius * std::cos(angle);
      double y = center.y + radius * std::sin(angle);
      points.push_back({x, y});
    }
    return points;
  };

  Jarvis::Point2d center{0, 0};
  double radius = 5.0;
  uint32_t numPoints = 100;  // Number of points for the circle

  std::vector<Jarvis::Point2d> points = generatePointsInCircle(center, radius, numPoints);
  // Expected result: the circle should be a convex hull
  std::vector<Jarvis::Point2d> expectedHull = points;  // Since the circle is already a convex hull

  std::vector<Jarvis::Point2d> out(points.size());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(points.size()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(out.size()));

  // Create Task
  Jarvis::JarvisTestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);

  // Such a strange check because my algorithm outputs points in a different order
  for (uint32_t i = 0; i < out.size(); ++i) {
    if (i < 50) {
      EXPECT_EQ(expectedHull[i].x, out[i + 50].x);
      EXPECT_EQ(expectedHull[i].y, out[i + 50].y);
    } else {
      EXPECT_EQ(expectedHull[i].x, out[i - 50].x);
      EXPECT_EQ(expectedHull[i].y, out[i - 50].y);
    }
  }
}
