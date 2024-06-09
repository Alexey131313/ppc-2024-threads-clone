// Copyright 2024 Filatov Maxim
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/filatov_m_linear_image_filtering/include/ops_seq.hpp"

TEST(filatov_m_linear_image_filtering, test_pipeline_run_with_blue_image) {
  uint32_t width = 1024;
  uint32_t height = 1024;

  // Create data
  std::vector<uint8_t> in(width * height * 3, 0);
  for (size_t i = 2; i < in.size(); i += 3) {
    in[i] = 255;  // Blue channel
  }
  std::vector<uint8_t> out(width * height * 3, 0);
  std::vector<uint8_t> expected = in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskImageData = std::make_shared<ppc::core::TaskData>();
  taskImageData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskImageData->inputs_count.emplace_back(width);
  taskImageData->inputs_count.emplace_back(height);
  taskImageData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskImageData->outputs_count.emplace_back(width);
  taskImageData->outputs_count.emplace_back(height);

  // Create Task
  auto gaussFilterHorizontal = std::make_shared<GaussFilterHorizontal>(taskImageData);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(gaussFilterHorizontal);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(expected, out);
}

TEST(filatov_m_linear_image_filtering, test_task_run_with_green_image) {
  uint32_t width = 1024;
  uint32_t height = 1024;

  // Create data
  std::vector<uint8_t> in(width * height * 3, 0);
  for (size_t i = 1; i < in.size(); i += 3) {
    in[i] = 255;  // Green channel
  }
  std::vector<uint8_t> out(width * height * 3, 0);
  std::vector<uint8_t> expected = in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskImageData = std::make_shared<ppc::core::TaskData>();
  taskImageData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskImageData->inputs_count.emplace_back(width);
  taskImageData->inputs_count.emplace_back(height);
  taskImageData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskImageData->outputs_count.emplace_back(width);
  taskImageData->outputs_count.emplace_back(height);

  // Create Task
  auto gaussFilterHorizontal = std::make_shared<GaussFilterHorizontal>(taskImageData);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(gaussFilterHorizontal);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(expected, out);
}
