// Copyright 2024 Pushkarev Ilya
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/pushkarev_i_dijkstra_shortest_path/include/dijkstra_shortest_path.hpp"

TEST(pushkarev_i_dijkstra_shortest_path_seq, test_pipeline_run) {
  const int count = 1000;

  std::vector<std::vector<int>> graph(count, std::vector<int>(count, 1));
  for (int i = 0; i < count; i++) {
    graph[i][i] = 0;
  }
  std::vector<int> distances(count, 0);
  int source = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&graph));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source));
  taskDataSeq->inputs_count.emplace_back(2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&distances));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  auto dijkstraTask = std::make_shared<DijkstraTask>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(dijkstraTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(distances[0], 0);
  ASSERT_EQ(distances[1], 1);
  ASSERT_EQ(distances[2], 1);
  ASSERT_EQ(distances[3], 1);
  ASSERT_EQ(distances[4], 1);
  ASSERT_EQ(distances[5], 1);
}

TEST(pushkarev_i_dijkstra_shortest_path_seq, test_task_run) {
  const int count = 2000;

  std::vector<std::vector<int>> graph(count, std::vector<int>(count, 1));
  for (int i = 0; i < count; i++) {
    graph[i][i] = 0;
  }
  std::vector<int> distances(count, 0);
  int source = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&graph));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source));
  taskDataSeq->inputs_count.emplace_back(2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&distances));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  auto dijkstraTask = std::make_shared<DijkstraTask>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(dijkstraTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(distances[0], 0);
  ASSERT_EQ(distances[1], 1);
  ASSERT_EQ(distances[2], 1);
  ASSERT_EQ(distances[3], 1);
  ASSERT_EQ(distances[4], 1);
  ASSERT_EQ(distances[5], 1);
}