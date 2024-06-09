// Copyright 2024 Ivlev Alexander
#include <gtest/gtest.h>

#include <vector>

#include "omp/ivlev_a_convex_hull/include/ops_omp.hpp"
using namespace ivlev_a_omp;

TEST(ivlev_a_convex_hull_omp, one_component) {
  // Create data
  std::vector<std::vector<std::pair<size_t, size_t>>> in = {{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 2}, {3, 3}}};
  std::vector<std::vector<std::pair<size_t, size_t>>> out_seq = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> out_par = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> res = {{{0, 0}, {0, 1}, {1, 0}, {3, 3}}};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs_count.emplace_back(in.size());
  out_par.resize(in.size());
  taskDataPar->outputs_count.emplace_back(out_par.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
  for (size_t i = 0; i < in.size(); i++) {
    taskDataPar->inputs_count.emplace_back(in[i].size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }

  // Create Task
  ConvexHullOMPTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  ASSERT_EQ(testTaskParallel.pre_processing(), true);
  ASSERT_EQ(testTaskParallel.run(), true);
  ASSERT_EQ(testTaskParallel.post_processing(), true);
  ASSERT_EQ(out_par, res);
}

TEST(ivlev_a_convex_hull_omp, two_component) {
  // Create data
  std::vector<std::vector<std::pair<size_t, size_t>>> in = {{{0, 0}}, {{0, 1}, {1, 0}}};
  std::vector<std::vector<std::pair<size_t, size_t>>> out_seq = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> out_par = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> res = in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs_count.emplace_back(in.size());
  out_par.resize(in.size());
  taskDataPar->outputs_count.emplace_back(out_par.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
  for (size_t i = 0; i < in.size(); i++) {
    taskDataPar->inputs_count.emplace_back(in[i].size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }

  // Create Task
  ConvexHullOMPTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  ASSERT_EQ(testTaskParallel.pre_processing(), true);
  ASSERT_EQ(testTaskParallel.run(), true);
  ASSERT_EQ(testTaskParallel.post_processing(), true);
  ASSERT_EQ(out_par, res);
}

TEST(ivlev_a_convex_hull_omp, three_component) {
  // Create data
  std::vector<std::vector<std::pair<size_t, size_t>>> in = {{{1, 1}}, {{0, 1}, {1, 0}}, {{0, 0}, {0, 1}, {1, 0}}};
  std::vector<std::vector<std::pair<size_t, size_t>>> out_seq = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> out_par = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> res = in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs_count.emplace_back(in.size());
  out_par.resize(in.size());
  taskDataPar->outputs_count.emplace_back(out_par.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
  for (size_t i = 0; i < in.size(); i++) {
    taskDataPar->inputs_count.emplace_back(in[i].size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }

  // Create Task
  ConvexHullOMPTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  ASSERT_EQ(testTaskParallel.pre_processing(), true);
  ASSERT_EQ(testTaskParallel.run(), true);
  ASSERT_EQ(testTaskParallel.post_processing(), true);
  ASSERT_EQ(out_par, res);
}

TEST(ivlev_a_convex_hull_omp, one_big_compnent) {
  // Create data
  std::vector<std::vector<std::pair<size_t, size_t>>> in = {
      {{0, 2}, {1, 1}, {1, 3}, {2, 0}, {2, 2}, {2, 4}, {3, 1}, {3, 3}, {4, 2}}};
  std::vector<std::vector<std::pair<size_t, size_t>>> out_seq = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> out_par = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> res = {{{0, 2}, {2, 0}, {2, 4}, {4, 2}}};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs_count.emplace_back(in.size());
  out_par.resize(in.size());
  taskDataPar->outputs_count.emplace_back(out_par.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
  for (size_t i = 0; i < in.size(); i++) {
    taskDataPar->inputs_count.emplace_back(in[i].size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }

  // Create Task
  ConvexHullOMPTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  ASSERT_EQ(testTaskParallel.pre_processing(), true);
  ASSERT_EQ(testTaskParallel.run(), true);
  ASSERT_EQ(testTaskParallel.post_processing(), true);
  ASSERT_EQ(out_par, res);
}

TEST(ivlev_a_convex_hull_omp, one_small_one_big) {
  // Create data
  std::vector<std::vector<std::pair<size_t, size_t>>> in = {
      {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
      {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 3}, {2, 0}, {2, 2}, {3, 0}, {3, 2}, {3, 3}}};
  std::vector<std::vector<std::pair<size_t, size_t>>> out_seq = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> out_par = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> res = {{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
                                                             {{0, 0}, {0, 2}, {1, 3}, {3, 0}, {3, 3}}};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs_count.emplace_back(in.size());
  out_par.resize(in.size());
  taskDataPar->outputs_count.emplace_back(out_par.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
  for (size_t i = 0; i < in.size(); i++) {
    taskDataPar->inputs_count.emplace_back(in[i].size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }

  // Create Task
  ConvexHullOMPTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  ASSERT_EQ(testTaskParallel.pre_processing(), true);
  ASSERT_EQ(testTaskParallel.run(), true);
  ASSERT_EQ(testTaskParallel.post_processing(), true);
  ASSERT_EQ(out_par, res);
}
