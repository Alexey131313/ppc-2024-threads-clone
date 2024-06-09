// Copyright 2024 Ionova Ekatetina
#include <gtest/gtest.h>
#include <omp.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/ionova_e_sparse_matr_multi_crs_complex_omp/include/ops_seq.hpp"

TEST(ionova_e_sparse_matr_multi_crs_complex_omp, test_pipeline_run) {
  size_t p = 500;
  size_t q = 500;
  size_t r = 500;
  std::vector<Complex> lhs_in(p * q);
  for (size_t i = 0; i < p; ++i) {
    if (i % 3 == 0)
      for (size_t j = 0; j < q; ++j) {
        lhs_in[i * q + j] = {1.0, 1.0};
      }
  }
  std::vector<Complex> rhs_in(q * r);
  for (size_t i = 0; i < q; ++i) {
    for (size_t k = 0; k < r; ++k) {
      if (k % 3 == 0) rhs_in[i * r + k] = {1.0, -1.0};
    }
  }
  std::vector<Complex> out(p * r);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lhs_in.data()));
  taskDataSeq->inputs_count.emplace_back(p);
  taskDataSeq->inputs_count.emplace_back(q);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs_in.data()));
  taskDataSeq->inputs_count.emplace_back(q);
  taskDataSeq->inputs_count.emplace_back(r);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(p);
  taskDataSeq->outputs_count.emplace_back(r);

  // Create Task
  auto testTaskOmp = std::make_shared<SparseMatrixComplexMultiSequentialOmp>(taskDataSeq);

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

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOmp);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (size_t i = 0; i < p; ++i) {
    for (size_t j = 0; j < r; ++j) {
      if (i % 3 == 0 && j % 3 == 0) {
        EXPECT_DOUBLE_EQ(out[i * r + j].real, 2 * q);
        EXPECT_DOUBLE_EQ(out[i * r + j].imag, 0.0);
      } else {
        EXPECT_DOUBLE_EQ(out[i * r + j].real, 0.0);
        EXPECT_DOUBLE_EQ(out[i * r + j].imag, 0.0);
      }
    }
  }
}

TEST(ionova_e_sparse_matr_multi_crs_complex_omp, test_task_run) {
  size_t p = 500;
  size_t q = 500;
  size_t r = 500;
  std::vector<Complex> lhs_in(p * q);
  for (size_t i = 0; i < p; ++i) {
    if (i % 3 == 0)
      for (size_t j = 0; j < q; ++j) {
        lhs_in[i * q + j] = {1.0, 1.0};
      }
  }
  std::vector<Complex> rhs_in(q * r);
  for (size_t i = 0; i < q; ++i) {
    for (size_t j = 0; j < r; ++j) {
      if (j % 3 == 0) rhs_in[i * r + j] = {1.0, -1.0};
    }
  }
  std::vector<Complex> out(p * r);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lhs_in.data()));
  taskDataSeq->inputs_count.emplace_back(p);
  taskDataSeq->inputs_count.emplace_back(q);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs_in.data()));
  taskDataSeq->inputs_count.emplace_back(q);
  taskDataSeq->inputs_count.emplace_back(r);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(p);
  taskDataSeq->outputs_count.emplace_back(r);

  // Create Task
  auto testTaskOmp = std::make_shared<SparseMatrixComplexMultiSequentialOmp>(taskDataSeq);

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

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOmp);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (size_t i = 0; i < p; ++i) {
    for (size_t j = 0; j < r; ++j) {
      if (i % 3 == 0 && j % 3 == 0) {
        EXPECT_DOUBLE_EQ(out[i * r + j].real, 2 * q);
        EXPECT_DOUBLE_EQ(out[i * r + j].imag, 0.0);
      } else {
        EXPECT_DOUBLE_EQ(out[i * r + j].real, 0.0);
        EXPECT_DOUBLE_EQ(out[i * r + j].imag, 0.0);
      }
    }
  }
}
