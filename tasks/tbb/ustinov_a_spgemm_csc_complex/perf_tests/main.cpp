// Copyright 2024 Ustinov Alexander
#include <gtest/gtest.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/tick_count.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "tbb/ustinov_a_spgemm_csc_complex/include/ops_tbb.hpp"

const double PI = 3.14159265358979323846;

sparse_matrix dft_matrix(int n) {
  auto N = (double)n;
  std::complex<double> exponent{0.0, -2.0 * PI / N};
  sparse_matrix dft(n, n, n * n);
  for (int i = 1; i <= n; ++i) {
    dft.col_ptr[i] = i * n;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      dft.rows[i * n + j] = j;
      dft.values[i * n + j] = std::exp(exponent * double(i * j));
    }
  }
  return dft;
}

sparse_matrix dft_conj_matrix(int n) {
  auto N = (double)n;
  std::complex<double> exponent{0.0, 2.0 * PI / N};
  sparse_matrix dft_conj(n, n, n * n);
  for (int i = 1; i <= n; ++i) {
    dft_conj.col_ptr[i] = i * n;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      dft_conj.rows[i * n + j] = j;
      dft_conj.values[i * n + j] = std::exp(exponent * double(j * i));
    }
  }
  return dft_conj;
}

TEST(ustinov_a_spgemm_csc_complex_omp, test_pipeline_run_dft384x384) {
  int n = 384;
  sparse_matrix A = dft_matrix(n);
  sparse_matrix B = dft_conj_matrix(n);
  sparse_matrix C;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&B));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C));

  // Create Task
  auto testTaskParallel = std::make_shared<SpgemmCSCComplexTBBPar>(taskDataPar);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(ustinov_a_spgemm_csc_complex_omp, test_task_run_dft384x384) {
  int n = 384;
  sparse_matrix A = dft_matrix(n);
  sparse_matrix B = dft_conj_matrix(n);
  sparse_matrix C;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&B));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C));

  // Create Task
  auto testTaskParallel = std::make_shared<SpgemmCSCComplexTBBPar>(taskDataPar);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}