// Copyright 2024 Moiseev Nikita
#include <gtest/gtest.h>

#include <vector>

#include "tbb/moiseev_n_jarvis_tbb/include/ops_tbb.hpp"

TEST(Tbb_Jarvis_Moiseev, Empty_Input) {
  std::vector<Point> points = {};
  std::vector<Point> hull = {};
  std::vector<Point> resHull(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataTbb->inputs_count.emplace_back(points.size());
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataTbb->outputs_count.emplace_back(resHull.size());

  // Create Task
  TestTaskTbbJarvisMoiseev testTaskTbb(taskDataTbb);
  ASSERT_EQ(testTaskTbb.validation(), false);
}

TEST(Tbb_Jarvis_Moiseev, Single_Point) {
  std::vector<Point> points = {{0, 0}};
  std::vector<Point> hull = {{0, 0}};
  std::vector<Point> resHull(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataTbb->inputs_count.emplace_back(points.size());
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataTbb->outputs_count.emplace_back(resHull.size());

  // Create Task
  TestTaskTbbJarvisMoiseev testTaskTbb(taskDataTbb);
  ASSERT_EQ(testTaskTbb.validation(), true);
  testTaskTbb.pre_processing();
  testTaskTbb.run();
  testTaskTbb.post_processing();

  ASSERT_EQ(resHull[0], hull[0]);
}

TEST(Tbb_Jarvis_Moiseev, Two_Points) {
  std::vector<Point> points = {{0, 0}, {1, 1}};
  std::vector<Point> hull = {{0, 0}, {1, 1}};
  std::vector<Point> resHull(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataTbb->inputs_count.emplace_back(points.size());
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataTbb->outputs_count.emplace_back(resHull.size());

  // Create Task
  TestTaskTbbJarvisMoiseev testTaskTbb(taskDataTbb);
  ASSERT_EQ(testTaskTbb.validation(), true);
  testTaskTbb.pre_processing();
  testTaskTbb.run();
  testTaskTbb.post_processing();

  ASSERT_EQ(resHull[0], hull[0]);
  ASSERT_EQ(resHull[1], hull[1]);
}

TEST(Tbb_Jarvis_Moiseev, Duplicate_Points) {
  std::vector<Point> points = {{0, 0}, {1, 1}, {2, 2}, {0, 0}, {1, 1}, {2, 2}};
  std::vector<Point> hull = {{0, 0}, {2, 2}};
  std::vector<Point> resHull(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataTbb->inputs_count.emplace_back(points.size());
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataTbb->outputs_count.emplace_back(resHull.size());
  // Create Task
  TestTaskTbbJarvisMoiseev testTaskTbb(taskDataTbb);
  ASSERT_EQ(testTaskTbb.validation(), true);
  testTaskTbb.pre_processing();
  testTaskTbb.run();
  testTaskTbb.post_processing();

  ASSERT_EQ(resHull[0], hull[0]);
  ASSERT_EQ(resHull[1], hull[1]);
}

TEST(Tbb_Jarvis_Moiseev, Random_Points) {
  std::vector<Point> points = {{1, 4}, {3, 8}, {8, 2}, {5, 5}, {9, 1}, {4, 7}};
  std::vector<Point> hull = {{1, 4}, {3, 8}, {4, 7}, {9, 1}};
  std::vector<Point> resHull(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataTbb->inputs_count.emplace_back(points.size());
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataTbb->outputs_count.emplace_back(resHull.size());

  // Create Task
  TestTaskTbbJarvisMoiseev testTaskTbb(taskDataTbb);
  ASSERT_EQ(testTaskTbb.validation(), true);
  testTaskTbb.pre_processing();
  testTaskTbb.run();
  testTaskTbb.post_processing();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(resHull[i], hull[i]);
  }
}

TEST(Tbb_Jarvis_Moiseev, Rectangle_Points) {
  std::vector<Point> points = {{0, 0}, {0, 5}, {5, 5}, {5, 0}, {1, 1}, {1, 4}, {4, 4}, {4, 1}};
  std::vector<Point> hull = {{0, 0}, {0, 5}, {5, 5}, {5, 0}};
  std::vector<Point> resHull(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataTbb->inputs_count.emplace_back(points.size());
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataTbb->outputs_count.emplace_back(resHull.size());

  // Create Task
  TestTaskTbbJarvisMoiseev testTaskTbb(taskDataTbb);
  ASSERT_EQ(testTaskTbb.validation(), true);
  testTaskTbb.pre_processing();
  testTaskTbb.run();
  testTaskTbb.post_processing();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(resHull[i], hull[i]);
  }
}
