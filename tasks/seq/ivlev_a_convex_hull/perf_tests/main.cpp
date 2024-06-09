// Copyright 2024 Ivlev Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/ivlev_a_convex_hull/include/ops_seq.hpp"
using namespace ivlev_a_seq;

TEST(sequential_ivlev_a_convex_hull_perf_test, test_pipeline_run_) {
  // Create data
  std::vector<std::vector<std::pair<size_t, size_t>>> in = {{}};
  in[0].emplace_back(0, 1);
  in[0].emplace_back(1, 0);
  in[0].emplace_back(1, 1);
  for (size_t i = 2; i < 2999999; i++) {
    in[0].emplace_back(i, i - 2);
    in[0].emplace_back(i, i - 1);
    in[0].emplace_back(i, i);
    in[0].emplace_back(i, i + 1);
    in[0].emplace_back(i, i + 2);
  }
  in[0].emplace_back(2999999, 2999999);
  in[0].emplace_back(2999999, 3000000);
  in[0].emplace_back(3000000, 2999999);
  std::vector<std::vector<std::pair<size_t, size_t>>> out = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> res = {
      {{0, 1}, {1, 0}, {2, 0}, {2, 4}, {2999998, 2999996}, {2999998, 3000000}, {2999999, 3000000}, {3000000, 2999999}}};

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
  auto testTaskSequential = std::make_shared<ConvexHullSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(out, res);
}

TEST(sequential_ivlev_a_convex_hull_perf_test, test_task_run_) {
  // Create data
  std::vector<std::vector<std::pair<size_t, size_t>>> in = {{}};
  in[0].emplace_back(0, 1);
  in[0].emplace_back(1, 0);
  in[0].emplace_back(1, 1);
  for (size_t i = 2; i < 2999999; i++) {
    in[0].emplace_back(i, i - 2);
    in[0].emplace_back(i, i - 1);
    in[0].emplace_back(i, i);
    in[0].emplace_back(i, i + 1);
    in[0].emplace_back(i, i + 2);
  }
  in[0].emplace_back(2999999, 2999999);
  in[0].emplace_back(2999999, 3000000);
  in[0].emplace_back(3000000, 2999999);
  std::vector<std::vector<std::pair<size_t, size_t>>> out = {};
  std::vector<std::vector<std::pair<size_t, size_t>>> res = {
      {{0, 1}, {1, 0}, {2, 0}, {2, 4}, {2999998, 2999996}, {2999998, 3000000}, {2999999, 3000000}, {3000000, 2999999}}};

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
  auto testTaskSequential = std::make_shared<ConvexHullSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(out, res);
}
