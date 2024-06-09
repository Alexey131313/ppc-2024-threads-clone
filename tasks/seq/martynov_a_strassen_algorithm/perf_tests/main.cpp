
// Copyright 2024 Martynov Aleksandr
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/martynov_a_strassen_algorithm/include/ops_seq.hpp"

TEST(martynov_a_strassen_alg_seq_perf, test_pipeline_run) {
  const int n = 64;
  int m = n * n;
  // Create data
  std::vector<double> first_matrix = fillMatrix(n);
  std::vector<double> second_matrix = fillMatrix(n);
  std::vector<double> result(m);
  std::vector<double> res = ijkalgorithm(first_matrix, second_matrix, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(first_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(first_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(second_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(second_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  // Create Task
  auto testTaskSequential = std::make_shared<Strssn_alg>(taskDataSeq);
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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < res.size(); i++) {
    ASSERT_DOUBLE_EQ(std::round(res[i]), std::round(result[i]));
  }
}

TEST(martynov_a_strassen_alg_seq_perf, test_task_run) {
  const int n = 64;
  int m = n * n;
  // Create data
  std::vector<double> result(m);
  std::vector<double> first_matrix = fillMatrix(n);
  std::vector<double> second_matrix = fillMatrix(n);
  std::vector<double> res = ijkalgorithm(first_matrix, second_matrix, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(first_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(first_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(second_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(second_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  // Create Task
  auto testTaskSequential = std::make_shared<Strssn_alg>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < res.size(); i++) {
    ASSERT_DOUBLE_EQ(std::round(res[i]), std::round(result[i]));
  }
}
