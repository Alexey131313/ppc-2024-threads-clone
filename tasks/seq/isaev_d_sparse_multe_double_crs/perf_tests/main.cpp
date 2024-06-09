// Copyright 2024 Isaev Dmitriy
#include <gtest/gtest.h>

#include <chrono>
#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/isaev_d_sparse_multe_double_crs/include/ops_seq.hpp"

TEST(isaev_d_sparse_multe_double_crs_seq, test_pipeline_run) {
  int N = 500;
  int M = 500;
  // Create data
  IsaevSeq::SparseMatrix A = IsaevSeq::getRandomMatrix(N, M, 0.50, 3200);
  IsaevSeq::SparseMatrix B = IsaevSeq::getRandomMatrix(M, N, 0.50, 1350);
  IsaevSeq::SparseMatrix C;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs_count.emplace_back(N);
  taskDataSeq->inputs_count.emplace_back(M);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->inputs_count.emplace_back(M);
  taskDataSeq->inputs_count.emplace_back(N);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C));
  taskDataSeq->outputs_count.emplace_back(M);
  taskDataSeq->outputs_count.emplace_back(N);

  // Create Task
  auto taskSequential = std::make_shared<IsaevSeq::SparseMultDoubleCRS>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(isaev_d_sparse_multe_double_crs_seq, test_task_run) {
  int N = 500;
  int M = 500;
  // Create data
  IsaevSeq::SparseMatrix A = IsaevSeq::getRandomMatrix(N, M, 0.50, 3200);
  IsaevSeq::SparseMatrix B = IsaevSeq::getRandomMatrix(M, N, 0.50, 1350);
  IsaevSeq::SparseMatrix C;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs_count.emplace_back(N);
  taskDataSeq->inputs_count.emplace_back(M);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->inputs_count.emplace_back(N);
  taskDataSeq->inputs_count.emplace_back(M);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C));
  taskDataSeq->outputs_count.emplace_back(M);
  taskDataSeq->outputs_count.emplace_back(N);

  // Create Task
  auto taskSequential = std::make_shared<IsaevSeq::SparseMultDoubleCRS>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
