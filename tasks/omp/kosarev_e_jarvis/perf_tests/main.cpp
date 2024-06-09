// Copyright 2024 Kosarev Egor
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/kosarev_e_jarvis/include/ops_omp.hpp"

TEST(kosarev_e_jarvis_hull_omp, test_pipeline_run) {
  std::vector<Kosarev_e_OMP_KosarevJarvisHull::Point> points =
      Kosarev_e_OMP_KosarevJarvisHull::generateRandomPoints(120000, -1400, 1400, -1400, 1400);
  std::vector<Kosarev_e_OMP_KosarevJarvisHull::Point> resHull = points;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resHull.data()));
  taskDataSeq->outputs_count.emplace_back(resHull.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<Kosarev_e_OMP_KosarevJarvisHull::TestOMPTaskParallelKosarevJarvisHull>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 100;
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
  for (const auto& hullPoint : resHull) {
    bool found = false;
    for (const auto& point : points) {
      if (hullPoint == point) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found);
  }
}

TEST(kosarev_e_jarvis_hull_omp, test_task_run) {
  std::vector<Kosarev_e_OMP_KosarevJarvisHull::Point> points =
      Kosarev_e_OMP_KosarevJarvisHull::generateRandomPoints(120000, -1400, 1400, -1400, 1400);
  std::vector<Kosarev_e_OMP_KosarevJarvisHull::Point> resHull = points;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resHull.data()));
  taskDataSeq->outputs_count.emplace_back(resHull.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<Kosarev_e_OMP_KosarevJarvisHull::TestOMPTaskParallelKosarevJarvisHull>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 100;
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
  for (const auto& hullPoint : resHull) {
    bool found = false;
    for (const auto& point : points) {
      if (hullPoint == point) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found);
  }
}