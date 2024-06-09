// Copyright Andrey Karagodin 2024
#include <gtest/gtest.h>

#include <seq/karagodin_a_dejkstra/include/dejkstra_seq.hpp>

#include "core/perf/include/perf.hpp"

TEST(karagodin_a_dejkstra_seq_perf_test, test_pipeline_run) {
  // Create data
  int entryNode = 0;
  int destNode = 999;
  int size = 1000;
  std::pair<std::vector<int>, int> result;
  std::vector<std::vector<int>> graphMap;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&entryNode));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&destNode));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&graphMap));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  // Create Task
  auto dejkstra_seq = std::make_shared<DejkstraTaskSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(dejkstra_seq);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(result, result);
}

TEST(karagodin_a_dejkstra_seq_perf_test, test_task_run) {
  // Create data
  int entryNode = 0;
  int destNode = 1199;
  int size = 1200;
  std::pair<std::vector<int>, int> result;
  std::vector<std::vector<int>> graphMap;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&entryNode));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&destNode));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&graphMap));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  // Create Task
  auto dejkstra_seq = std::make_shared<DejkstraTaskSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(dejkstra_seq);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(result, result);
}
