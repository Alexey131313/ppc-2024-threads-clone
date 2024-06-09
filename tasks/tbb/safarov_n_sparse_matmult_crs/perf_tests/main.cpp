// Copyright 2024 Safarov Nurlan
#include <gtest/gtest.h>
#include <tbb/tbb.h>

#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "tbb/safarov_n_sparse_matmult_crs/include/sparse_matmult_crs_tbb.hpp"

TEST(Safarov_N_SparseMatMultCRS_TBB, test_pipeline_run) {
  SparseMatrixCRS X = SparseMatrixCRS(createRandomMatrix(235, 235, 0.80));
  SparseMatrixCRS Y = SparseMatrixCRS(createRandomMatrix(235, 235, 0.80));
  SparseMatrixCRS Z;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&X));
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&Y));
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Z));

  // Create Task
  auto taskTbb = std::make_shared<SparseMatrixMultiplicationCRS_TBB>(taskDataTbb);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 50;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskTbb);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(Safarov_N_SparseMatMultCRS_TBB, test_task_run) {
  SparseMatrixCRS X = SparseMatrixCRS(createRandomMatrix(235, 235, 0.80));
  SparseMatrixCRS Y = SparseMatrixCRS(createRandomMatrix(235, 235, 0.80));
  SparseMatrixCRS Z;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&X));
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&Y));
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Z));

  // Create Task
  auto taskTbb = std::make_shared<SparseMatrixMultiplicationCRS_TBB>(taskDataTbb);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 50;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskTbb);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}