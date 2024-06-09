// Copyright Andrey Karagodin 2024
#include <gtest/gtest.h>
#include <oneapi/tbb.h>

#include <tbb/karagodin_a_dejkstra/include/dejkstra_tbb.hpp>

#include "core/perf/include/perf.hpp"

TEST(karagodin_a_dejkstra_tbb_perf_test, test_pipeline_run) {
  // Create data
  int entryNode = 0;
  int destNode = 499;
  int size = 500;
  std::pair<std::vector<int>, int> result;
  std::vector<std::vector<int>> graphMap;
  // Create TaskData
  auto taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&entryNode));
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&destNode));
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&graphMap));
  taskDataTbb->inputs_count.emplace_back(size);
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  // Create Task
  auto dejkstra_tbb = std::make_shared<DejkstraTaskTBB>(taskDataTbb);

  // Create Perf attributes
  const auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  const auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  const auto perfAnalyzer = std::make_shared<ppc::core::Perf>(dejkstra_tbb);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(result, result);
}

TEST(karagodin_a_dejkstra_tbb_perf_test, test_task_run) {
  // Create data
  int entryNode = 0;
  int destNode = 799;
  int size = 800;
  std::pair<std::vector<int>, int> result;
  std::vector<std::vector<int>> graphMap;
  // Create TaskData
  auto taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&entryNode));
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&destNode));
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&graphMap));
  taskDataTbb->inputs_count.emplace_back(size);
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  // Create Task
  auto dejkstra_tbb = std::make_shared<DejkstraTaskTBB>(taskDataTbb);

  // Create Perf attributes
  const auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  const auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  const auto perfAnalyzer = std::make_shared<ppc::core::Perf>(dejkstra_tbb);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(result, result);
}
