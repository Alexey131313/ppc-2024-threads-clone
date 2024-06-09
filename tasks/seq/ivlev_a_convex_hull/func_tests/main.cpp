// Copyright 2024 Ivlev Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/ivlev_a_convex_hull/include/ops_seq.hpp"
using namespace ivlev_a_seq;

TEST(ivlev_a_convex_hull_sequential, one_component) {
  // Create data
  std::vector<std::vector<std::pair<size_t, size_t>>> in = {{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 2}, {3, 3}}};
  std::vector<std::vector<std::pair<size_t, size_t>>> out = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> res = {{{0, 0}, {0, 1}, {1, 0}, {3, 3}}};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(in.size());
  out.resize(in.size());
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  for (size_t i = 0; i < in.size(); i++) {
    taskDataSeq->inputs_count.emplace_back(in[i].size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }

  // Create Task
  ConvexHullSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, res);
}

TEST(ivlev_a_convex_hull_sequential, two_component) {
  // Create data
  std::vector<std::vector<std::pair<size_t, size_t>>> in = {{{0, 0}}, {{0, 1}, {1, 0}}};
  std::vector<std::vector<std::pair<size_t, size_t>>> out = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> res = in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(in.size());
  out.resize(in.size());
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  for (size_t i = 0; i < in.size(); i++) {
    taskDataSeq->inputs_count.emplace_back(in[i].size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }

  // Create Task
  ConvexHullSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, res);
}

TEST(ivlev_a_convex_hull_sequential, three_component) {
  // Create data
  std::vector<std::vector<std::pair<size_t, size_t>>> in = {{{1, 1}}, {{0, 1}, {1, 0}}, {{0, 0}, {0, 1}, {1, 0}}};
  std::vector<std::vector<std::pair<size_t, size_t>>> out = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> res = in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(in.size());
  out.resize(in.size());
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  for (size_t i = 0; i < in.size(); i++) {
    taskDataSeq->inputs_count.emplace_back(in[i].size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }

  // Create Task
  ConvexHullSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, res);
}

TEST(ivlev_a_convex_hull_sequential, one_big_compnent) {
  // Create data
  std::vector<std::vector<std::pair<size_t, size_t>>> in = {
      {{0, 2}, {1, 1}, {1, 3}, {2, 0}, {2, 2}, {2, 4}, {3, 1}, {3, 3}, {4, 2}}};
  std::vector<std::vector<std::pair<size_t, size_t>>> out = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> res = {{{0, 2}, {2, 0}, {2, 4}, {4, 2}}};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(in.size());
  out.resize(in.size());
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  for (size_t i = 0; i < in.size(); i++) {
    taskDataSeq->inputs_count.emplace_back(in[i].size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }

  // Create Task
  ConvexHullSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, res);
}

TEST(ivlev_a_convex_hull_sequential, one_small_one_big) {
  // Create data
  std::vector<std::vector<std::pair<size_t, size_t>>> in = {
      {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
      {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 3}, {2, 0}, {2, 2}, {3, 0}, {3, 2}, {3, 3}}};
  std::vector<std::vector<std::pair<size_t, size_t>>> out = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> res = {{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
                                                             {{0, 0}, {0, 2}, {1, 3}, {3, 0}, {3, 3}}};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(in.size());
  out.resize(in.size());
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  for (size_t i = 0; i < in.size(); i++) {
    taskDataSeq->inputs_count.emplace_back(in[i].size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }

  // Create Task
  ConvexHullSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, res);
}
