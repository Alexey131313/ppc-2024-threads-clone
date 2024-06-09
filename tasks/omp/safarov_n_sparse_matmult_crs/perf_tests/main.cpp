// Copyright 2024 Safarov Nurlan
#include <gtest/gtest.h>
#include <omp.h>

#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/safarov_n_sparse_matmult_crs/include/sparse_matmult_crs_omp.hpp"

TEST(Safarov_N_SparseMatMultCRS_OMP, test_pipeline_run) {
  SparseMatrixCRS X = SparseMatrixCRS(createRandomMatrix(235, 235, 0.80));
  SparseMatrixCRS Y = SparseMatrixCRS(createRandomMatrix(235, 235, 0.80));
  SparseMatrixCRS Z;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&X));
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&Y));
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Z));

  // Create Task
  auto taskOmp = std::make_shared<SparseMatrixMultiplicationCRS_OMP>(taskDataOmp);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = omp_get_wtime();
  perfAttr->current_timer = [&] { return omp_get_wtime() - t0; };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskOmp);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(Safarov_N_SparseMatMultCRS_OMP, test_task_run) {
  SparseMatrixCRS X = SparseMatrixCRS(createRandomMatrix(235, 235, 0.80));
  SparseMatrixCRS Y = SparseMatrixCRS(createRandomMatrix(235, 235, 0.80));
  SparseMatrixCRS Z;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&X));
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&Y));
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Z));

  // Create Task
  auto taskOmp = std::make_shared<SparseMatrixMultiplicationCRS_OMP>(taskDataOmp);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = omp_get_wtime();
  perfAttr->current_timer = [&] { return omp_get_wtime() - t0; };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskOmp);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}