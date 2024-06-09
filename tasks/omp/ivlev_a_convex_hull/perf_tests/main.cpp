// Copyright 2024 Ivlev Alexander
#include <gtest/gtest.h>
#include <omp.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/ivlev_a_convex_hull/include/ops_omp.hpp"
using namespace ivlev_a_omp;

TEST(ivlev_a_convex_hull_perf_test_omp, test_pipeline_run_omp) {
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
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs_count.emplace_back(in.size());
  out.resize(in.size());
  taskDataPar->outputs_count.emplace_back(out.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  for (size_t i = 0; i < in.size(); i++) {
    taskDataPar->inputs_count.emplace_back(in[i].size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }

  // Create Task
  auto testTaskParallel = std::make_shared<ConvexHullOMPTaskParallel>(taskDataPar);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(out, res);
}

TEST(ivlev_a_convex_hull_perf_test_omp, test_task_run_omp) {
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
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs_count.emplace_back(in.size());
  out.resize(in.size());
  taskDataPar->outputs_count.emplace_back(out.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  for (size_t i = 0; i < in.size(); i++) {
    taskDataPar->inputs_count.emplace_back(in[i].size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }

  // Create Task
  auto testTaskParallel = std::make_shared<ConvexHullOMPTaskParallel>(taskDataPar);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(out, res);
}
