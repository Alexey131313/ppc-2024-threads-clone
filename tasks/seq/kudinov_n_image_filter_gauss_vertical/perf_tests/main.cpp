// Copyright 2024 Kudinov Nikita
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kudinov_n_image_filter_gauss_vertical/include/image_filter_gauss_vertical_task.hpp"

const int seed = 124913;
const std::size_t expectedImageHash = 4374989887818936358;

TEST(kudinov_n_image_filter_gauss_vertical_seq, test_pipeline_run) {
  // Create data
  std::size_t radius = 3;
  double sigma = 1.6;
  auto gauss_kernel = GaussKernel(radius, sigma);

  std::size_t height = 1000;
  std::size_t width = 800;

  std::mt19937 gen(seed);
  std::uniform_int_distribution<> dis(0, UINT8_MAX);
  std::vector<Pixel> pixels;
  pixels.reserve(height * width);
  for (std::size_t i = 0; i < height * width; i++) {
    pixels.emplace_back(dis(gen));
  }

  auto input_image = Image(height, width, pixels);
  auto output_image = Image(height, width, std::vector<Pixel>(height * width, 0));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&gauss_kernel));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_image));
  taskData->inputs_count.emplace_back(1);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&output_image));
  taskData->outputs_count.emplace_back(1);

  // Create Task
  auto task = std::make_shared<ImageFilterGaussVerticalTask>(taskData);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(output_image.hash(), expectedImageHash);
}

TEST(kudinov_n_image_filter_gauss_vertical_seq, test_task_run) {
  // Create data
  std::size_t radius = 3;
  double sigma = 1.6;
  auto gauss_kernel = GaussKernel(radius, sigma);

  std::size_t height = 1000;
  std::size_t width = 800;

  std::mt19937 gen(seed);
  std::uniform_int_distribution<> dis(0, UINT8_MAX);
  std::vector<Pixel> pixels;
  pixels.reserve(height * width);
  for (std::size_t i = 0; i < height * width; i++) {
    pixels.emplace_back(dis(gen));
  }

  auto input_image = Image(height, width, pixels);
  auto output_image = Image(height, width, std::vector<Pixel>(height * width, 0));

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&gauss_kernel));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_image));
  taskData->inputs_count.emplace_back(1);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&output_image));
  taskData->outputs_count.emplace_back(1);

  // Create Task
  auto task = std::make_shared<ImageFilterGaussVerticalTask>(taskData);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(output_image.hash(), expectedImageHash);
}
