// Copyright 2024 Skotin Alexander
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/skotin_a_multiply_matrix_cannon/include/ops_seq.hpp"

std::vector<uint8_t> createMatrixData(size_t size, double value) {
  std::vector<double> matrixData(size * size, value);
  std::vector<uint8_t> byteData(matrixData.size() * sizeof(double));
  memcpy(byteData.data(), matrixData.data(), byteData.size());
  return byteData;
}

void checkMatrixMultiplicationResult(const std::vector<uint8_t>& outputData, size_t matrixSize, double expectedValue) {
  for (size_t i = 0; i < matrixSize; ++i) {
    for (size_t j = 0; j < matrixSize; ++j) {
      double actualValue;
      memcpy(&actualValue, &outputData[(i * matrixSize + j) * sizeof(double)], sizeof(double));
      EXPECT_NEAR(actualValue, expectedValue, 1e-8) << "Mismatch at (" << i << "," << j << ")";
    }
  }
}

TEST(Skotin_A_Multiply_Matrix_Cannon_Seq, test_pipeline_run) {
  const size_t matrixSize = 500;
  auto matrixAData = createMatrixData(matrixSize, 1.0);
  auto matrixBData = createMatrixData(matrixSize, 2.0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(matrixAData.data());
  taskData->inputs_count.push_back(matrixAData.size());
  taskData->inputs.push_back(matrixBData.data());
  taskData->inputs_count.push_back(matrixBData.size());

  std::vector<uint8_t> outputData(matrixSize * matrixSize * sizeof(double));
  taskData->outputs.push_back(outputData.data());
  taskData->outputs_count.push_back(outputData.size());

  auto task = std::make_shared<MatrixMultiplicationTask>(taskData);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();

  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  checkMatrixMultiplicationResult(outputData, matrixSize, 10000.0);
}

TEST(Skotin_A_Multiply_Matrix_Cannon_Seq, test_task_run) {
  const size_t matrixSize = 500;
  auto matrixAData = createMatrixData(matrixSize, 1.0);
  auto matrixBData = createMatrixData(matrixSize, 2.0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(matrixAData.data());
  taskData->inputs_count.push_back(matrixAData.size());
  taskData->inputs.push_back(matrixBData.data());
  taskData->inputs_count.push_back(matrixBData.size());

  std::vector<uint8_t> outputData(matrixSize * matrixSize * sizeof(double));
  taskData->outputs.push_back(outputData.data());
  taskData->outputs_count.push_back(outputData.size());

  auto task = std::make_shared<MatrixMultiplicationTask>(taskData);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();

  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  checkMatrixMultiplicationResult(outputData, matrixSize, 11000.0);
}
