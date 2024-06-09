// Copyright 2024 Shipilov Artem
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/shipilov_dijkstra/include/shipilov_a_ops_seq.hpp"

using namespace Dijkstra;

TEST(Dijkstra_Shipilov_seq_test, test_pipeline_run) {
  // Create data
  constexpr size_t n = 4000;

  graph_t graph(n);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
      int w = i + j;
      graph.add_edge(i, edge_t(j, w));
      graph.add_edge(j, edge_t(i, w));
    }
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  // Create Task
  auto testTaskSequential = std::make_shared<Dijkstra::MyDijkstra>(taskDataSeq);
  testTaskSequential->graph = std::move(graph);

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
}

TEST(Dijkstra_Shipilov_seq_test, test_task_run) {
  // Create data
  constexpr size_t n = 4000;

  graph_t graph(n);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
      int w = i + j;
      graph.add_edge(i, edge_t(j, w));
      graph.add_edge(j, edge_t(i, w));
    }
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  // Create Task
  auto testTaskSequential = std::make_shared<Dijkstra::MyDijkstra>(taskDataSeq);
  testTaskSequential->graph = std::move(graph);

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
}
