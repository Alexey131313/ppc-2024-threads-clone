// Copyright 2024 Vanushkin Dmitry
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/vanushkin_d_sobel_operator/include/sobel_operator_omp.hpp"

TEST(Vanushkin_D_OmpParallelSobelOperator, test_task_pipeline) {
  size_t width = 1024;
  size_t height = 960;

  std::vector<Color> in(width * height, Color(0));
  std::vector<Grayscale> out((width - 2) * (height - 2), 0);

  std::vector<Grayscale> expected((width - 2) * (height - 2), 0);

  // Create task data
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width - 2);
  taskDataSeq->outputs_count.emplace_back(height - 2);

  // Crete task
  auto sobelOperatorTask = std::make_shared<SobelOperatorParallelOmp>(taskDataSeq);

  // Create perf attribute
  auto perfAttribute = std::make_shared<ppc::core::PerfAttr>();

  const auto t0 = std::chrono::high_resolution_clock::now();

  perfAttribute->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  perfAttribute->num_running = 10;

  // Create perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sobelOperatorTask);
  perfAnalyzer->pipeline_run(perfAttribute, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(out[i], expected[i]);
  }
}

TEST(Vanushkin_D_OmpParallelSobelOperator, test_task_run) {
  size_t width = 1024;
  size_t height = 960;

  std::vector<Color> in(width * height, Color(0));
  std::vector<Grayscale> out((width - 2) * (height - 2), 0);

  std::vector<Grayscale> expected((width - 2) * (height - 2), 0);

  // Create task data
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width - 2);
  taskDataSeq->outputs_count.emplace_back(height - 2);

  // Crete task
  auto sobelOperatorTask = std::make_shared<SobelOperatorParallelOmp>(taskDataSeq);

  // Create perf attribute
  auto perfAttribute = std::make_shared<ppc::core::PerfAttr>();

  const auto t0 = std::chrono::high_resolution_clock::now();

  perfAttribute->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  perfAttribute->num_running = 9;

  // Create perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sobelOperatorTask);
  perfAnalyzer->task_run(perfAttribute, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(out[i], expected[i]);
  }
}
