// Copyright 2024 Borisov Saveliy
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "stl/borisov_s_convex_hull/include/ops_stl.hpp"

TEST(borisov_s_convex_hull_stl, test_pipeline_run) {
  size_t width = 2000;
  size_t height = 3;

  // Create data
  std::vector<uint8_t> in(height * width, 0);
  for (size_t j = 0; j < width; ++j) {
    in[j] = 1;
    in[(height - 1) * width + j] = 1;
  }
  for (size_t i = 0; i < height; ++i) {
    in[i * width] = 1;
    in[i * width + width - 1] = 1;
  }
  std::vector<uint8_t> out(width * height, 0);
  std::vector<uint8_t> expected(width * height, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  auto convexHull = std::make_shared<BorisovSaveliySTL::ConvexHull>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(convexHull);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(expected, out);
}

TEST(borisov_s_convex_hull_stl, test_task_run) {
  size_t width = 2000;
  size_t height = 3;

  // Create data
  std::vector<uint8_t> in(height * width, 0);
  for (size_t j = 0; j < width; ++j) {
    in[j] = 1;
    in[(height - 1) * width + j] = 1;
  }
  for (size_t i = 0; i < height; ++i) {
    in[i * width] = 1;
    in[i * width + width - 1] = 1;
  }
  std::vector<uint8_t> out(width * height, 0);
  std::vector<uint8_t> expected(width * height, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  auto convexHull = std::make_shared<BorisovSaveliySTL::ConvexHull>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(convexHull);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(expected, out);
}