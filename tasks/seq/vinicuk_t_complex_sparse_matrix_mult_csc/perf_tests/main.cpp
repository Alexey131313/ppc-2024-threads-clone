// Copyright 2024 Vinichuk Timofey
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vinicuk_t_complex_sparse_matrix_mult_csc/include/csc_complex_matrix.hpp"
#include "seq/vinicuk_t_complex_sparse_matrix_mult_csc/include/ops_seq.hpp"

TEST(vinicuk_t_complex_sparse_matrix_mult_csc_seq, test_pipeline_run_dft384x384) {
  size_t n = 500;
  CSCComplexMatrix mtrx_A(n, n);
  CSCComplexMatrix mtrx_B(1, n);
  CSCComplexMatrix mtrx_res(1, n);
  std::complex<double> a(2.0, 1.0);

  int k = 1;
  int p = 0;
  for (size_t i = 0; i <= n; i++) {
    mtrx_A.col_ptrs.push_back(p);
    p += k;
    k++;
  }

  k = 1;
  p = 0;
  for (size_t i = 0; i < mtrx_A.col_ptrs[n]; i++) {
    mtrx_A.values.emplace_back(a);
    if (p >= k) {
      p = 0;
      k++;
    }
    mtrx_A.row_indexes.push_back(p);
    p++;
  }

  mtrx_B.col_ptrs = {0, n};
  for (size_t i = 0; i < n; i++) {
    mtrx_B.values.emplace_back(a);
    mtrx_B.row_indexes.push_back(i);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_res));

  // Create Task
  auto testTaskSequential = std::make_shared<MultMatrixCSCComplex>(taskDataSeq);

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
}

TEST(vinicuk_t_complex_sparse_matrix_mult_csc_seq, test_task_run_dft384x384) {
  size_t n = 500;
  CSCComplexMatrix mtrx_A(n, n);
  CSCComplexMatrix mtrx_B(1, n);
  CSCComplexMatrix mtrx_res(1, n);
  std::complex<double> a(2.0, 1.0);

  int k = 1;
  int p = 0;
  for (size_t i = 0; i <= n; i++) {
    mtrx_A.col_ptrs.emplace_back(p);
    p += k;
    k++;
  }
  // mtrx_A.col_ptrs = {0,1,3,6, 10, 15, 21};

  k = 1;
  p = 0;
  for (size_t i = 0; i < mtrx_A.col_ptrs[n]; i++) {
    mtrx_A.values.emplace_back(a);
    if (p >= k) {
      p = 0;
      k++;
    }
    mtrx_A.row_indexes.push_back(p);
    p++;
  }

  mtrx_B.col_ptrs = {0, n};
  for (size_t i = 0; i < n; i++) {
    mtrx_B.values.emplace_back(a);
    mtrx_B.row_indexes.push_back(i);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_res));

  // Create Task
  auto testTaskSequential = std::make_shared<MultMatrixCSCComplex>(taskDataSeq);

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
}
