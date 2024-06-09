// Copyright 2024 Borisov Saveliy
#include <gtest/gtest.h>

#include <vector>

#include "seq/borisov_s_convex_hull/include/ops_seq.hpp"

TEST(borisov_s_convex_hull_sequential, no_point) {
  uint32_t width = 8;
  uint32_t height = 8;

  // Create data
  std::vector<uint8_t> in(width * height, 0);
  std::vector<uint8_t> out(width * height, 0);
  std::vector<uint8_t> expected(width * height, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  BorisovSaveliySEQ::ConvexHullSequential convexHullSequential(taskDataSeq);
  ASSERT_EQ(convexHullSequential.validation(), true);
  convexHullSequential.pre_processing();
  convexHullSequential.run();
  convexHullSequential.post_processing();
  ASSERT_EQ(expected, out);
}
TEST(borisov_s_convex_hull_sequential, single_point) {
  uint32_t width = 3;
  uint32_t height = 3;

  // Create data
  std::vector<uint8_t> in = {0, 0, 0, 0, 1, 0, 0, 0, 0};
  std::vector<uint8_t> out(width * height, 0);
  std::vector<uint8_t> expected = {0, 0, 0, 0, 1, 0, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  BorisovSaveliySEQ::ConvexHullSequential convexHullSequential(taskDataSeq);
  ASSERT_EQ(convexHullSequential.validation(), true);
  convexHullSequential.pre_processing();
  convexHullSequential.run();
  convexHullSequential.post_processing();
  ASSERT_EQ(expected, out);
}
TEST(borisov_s_convex_hull_sequential, two_points) {
  uint32_t width = 3;
  uint32_t height = 3;

  // Create data
  std::vector<uint8_t> in = {0, 1, 0, 0, 1, 0, 0, 0, 0};
  std::vector<uint8_t> out(width * height, 0);
  std::vector<uint8_t> expected = {0, 1, 0, 0, 1, 0, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  BorisovSaveliySEQ::ConvexHullSequential convexHullSequential(taskDataSeq);
  ASSERT_EQ(convexHullSequential.validation(), true);
  convexHullSequential.pre_processing();
  convexHullSequential.run();
  convexHullSequential.post_processing();
  ASSERT_EQ(expected, out);
}
TEST(borisov_s_convex_hull_sequential, corner_to_triangle) {
  uint32_t width = 9;
  uint32_t height = 6;

  // Create data
  std::vector<uint8_t> in = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint8_t> out(width * height, 0);
  std::vector<uint8_t> expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                                   0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  BorisovSaveliySEQ::ConvexHullSequential convexHullSequential(taskDataSeq);
  ASSERT_EQ(convexHullSequential.validation(), true);
  convexHullSequential.pre_processing();
  convexHullSequential.run();
  convexHullSequential.post_processing();
  ASSERT_EQ(expected, out);
}
TEST(borisov_s_convex_hull_sequential, frame_to_square) {
  uint32_t width = 9;
  uint32_t height = 6;

  // Create data
  std::vector<uint8_t> in = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint8_t> out(width * height, 0);
  std::vector<uint8_t> expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
                                   0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  BorisovSaveliySEQ::ConvexHullSequential convexHullSequential(taskDataSeq);
  ASSERT_EQ(convexHullSequential.validation(), true);
  convexHullSequential.pre_processing();
  convexHullSequential.run();
  convexHullSequential.post_processing();
  ASSERT_EQ(expected, out);
}
TEST(borisov_s_convex_hull_sequential, cross_to_star) {
  uint32_t width = 9;
  uint32_t height = 9;

  // Create data
  std::vector<uint8_t> in = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint8_t> out(width * height, 0);
  std::vector<uint8_t> expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                                   0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
                                   0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  BorisovSaveliySEQ::ConvexHullSequential convexHullSequential(taskDataSeq);
  ASSERT_EQ(convexHullSequential.validation(), true);
  convexHullSequential.pre_processing();
  convexHullSequential.run();
  convexHullSequential.post_processing();
  ASSERT_EQ(expected, out);
}
