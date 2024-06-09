// Copyright 2024 Isaev Dmitriy
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/isaev_d_sparse_multe_double_crs/include/ops_omp.hpp"

TEST(isaev_d_sparse_multe_double_crs_omp, test_pipeline_run) {
  int N = 500;
  int M = 500;

  // Create data
  IsaevOMP::SparseMatrix A = IsaevOMP::getRandomMatrix(N, M, 0.50, 3200);
  IsaevOMP::SparseMatrix B = IsaevOMP::getRandomMatrix(M, N, 0.50, 1350);
  IsaevOMP::SparseMatrix C;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataOMP = std::make_shared<ppc::core::TaskData>();
  taskDataOMP->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataOMP->inputs_count.emplace_back(N);
  taskDataOMP->inputs_count.emplace_back(M);
  taskDataOMP->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataOMP->inputs_count.emplace_back(M);
  taskDataOMP->inputs_count.emplace_back(N);
  taskDataOMP->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C));
  taskDataOMP->outputs_count.emplace_back(M);
  taskDataOMP->outputs_count.emplace_back(N);

  // Create Task
  auto taskOMP = std::make_shared<IsaevOMP::SparseMultDoubleCRSompParallel>(taskDataOMP);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskOMP);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(isaev_d_sparse_multe_double_crs_omp, test_task_run) {
  int N = 500;
  int M = 500;
  // Create data
  IsaevOMP::SparseMatrix A = IsaevOMP::getRandomMatrix(N, M, 0.50, 3200);
  IsaevOMP::SparseMatrix B = IsaevOMP::getRandomMatrix(M, N, 0.50, 1350);
  IsaevOMP::SparseMatrix C;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataOMP = std::make_shared<ppc::core::TaskData>();
  taskDataOMP->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataOMP->inputs_count.emplace_back(N);
  taskDataOMP->inputs_count.emplace_back(M);
  taskDataOMP->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataOMP->inputs_count.emplace_back(N);
  taskDataOMP->inputs_count.emplace_back(M);
  taskDataOMP->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C));
  taskDataOMP->outputs_count.emplace_back(M);
  taskDataOMP->outputs_count.emplace_back(N);

  // Create Task
  auto taskOMP = std::make_shared<IsaevOMP::SparseMultDoubleCRSompParallel>(taskDataOMP);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskOMP);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
