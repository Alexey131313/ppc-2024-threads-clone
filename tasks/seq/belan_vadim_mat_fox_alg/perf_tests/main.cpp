// Copyright 2023 Belan Vadim
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/belan_vadim_mat_fox_alg/include/ops_seq.hpp"

TEST(matrix_mul_fox_perf_test, test_pipeline_run) {
  const int n = 1024;
  const int block_size = 16;

  std::vector<int> input(2, 0);
  input[0] = n;
  input[1] = block_size;

  std::vector<double> matrix_a(n * n, 1.0);
  std::vector<double> matrix_b(n * n, 2.0);
  std::vector<double> matrix_c(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  taskData->inputs_count.emplace_back(matrix_a.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  taskData->inputs_count.emplace_back(matrix_b.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  taskData->outputs_count.emplace_back(matrix_c.size());

  auto matrixMulFox = std::make_shared<MatrixMulFox>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(matrixMulFox);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(matrix_mul_fox_perf_test, test_task_run) {
  const int n = 1024;
  const int block_size = 16;

  std::vector<int> input(2, 0);
  input[0] = n;
  input[1] = block_size;

  std::vector<double> matrix_a(n * n, 1.0);
  std::vector<double> matrix_b(n * n, 2.0);
  std::vector<double> matrix_c(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  taskData->inputs_count.emplace_back(matrix_a.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  taskData->inputs_count.emplace_back(matrix_b.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  taskData->outputs_count.emplace_back(matrix_c.size());

  auto matrixMulFox = std::make_shared<MatrixMulFox>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(matrixMulFox);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
