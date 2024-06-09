// Copyright 2024 Moiseev Nikita
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "tbb/moiseev_n_jarvis_tbb/include/ops_tbb.hpp"

TEST(Tbb_Jarvis_Perf_Moiseev, test_pipeline_run) {
  std::vector<Point> points;
  const int size = 666'666;
  points.reserve(size);
  for (int i = 0; i < size; i++) {
    points.push_back({i % 100, i % 200});
  }
  std::vector<Point> resHull(points.size());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataTbb->inputs_count.emplace_back(points.size());
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataTbb->outputs_count.emplace_back(resHull.size());

  // Create Task
  auto testTaskTbb = std::make_shared<TestTaskTbbJarvisMoiseev>(taskDataTbb);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskTbb);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(Tbb_Jarvis_Perf_Moiseev, test_task_run) {
  std::vector<Point> points;
  const int size = 666'666;
  points.reserve(size);
  for (int i = 0; i < size; i++) {
    points.push_back({rand() % 100, rand() % 200});
  }

  std::vector<Point> resHull(points.size());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataTbb->inputs_count.emplace_back(points.size());
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataTbb->outputs_count.emplace_back(resHull.size());

  // Create Task
  auto testTaskTbb = std::make_shared<TestTaskTbbJarvisMoiseev>(taskDataTbb);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskTbb);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
