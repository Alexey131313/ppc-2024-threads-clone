// Copyright 2024 Safarov Nurlan
#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <thread>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "stl/safarov_n_sparse_matmult_crs/include/sparse_matmult_crs_stl.hpp"

TEST(Safarov_N_SparseMatMultCRS_STL, test_pipeline_run) {
  SparseMatrixCRS X = SparseMatrixCRS(createRandomMatrix(235, 235, 0.80));
  SparseMatrixCRS Y = SparseMatrixCRS(createRandomMatrix(235, 235, 0.80));
  SparseMatrixCRS Z;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataStl = std::make_shared<ppc::core::TaskData>();
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(&X));
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(&Y));
  taskDataStl->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Z));

  // Create Task
  auto taskStl = std::make_shared<SparseMatrixMultiplicationCRS_STL>(taskDataStl);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(current_time_point - t0).count();
    return duration;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskStl);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(Safarov_N_SparseMatMultCRS_STL, test_task_run) {
  SparseMatrixCRS X = SparseMatrixCRS(createRandomMatrix(235, 235, 0.80));
  SparseMatrixCRS Y = SparseMatrixCRS(createRandomMatrix(235, 235, 0.80));
  SparseMatrixCRS Z;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataStl = std::make_shared<ppc::core::TaskData>();
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(&X));
  taskDataStl->inputs.emplace_back(reinterpret_cast<uint8_t *>(&Y));
  taskDataStl->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Z));

  // Create Task
  auto taskStl = std::make_shared<SparseMatrixMultiplicationCRS_STL>(taskDataStl);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(current_time_point - t0).count();
    return duration;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskStl);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}