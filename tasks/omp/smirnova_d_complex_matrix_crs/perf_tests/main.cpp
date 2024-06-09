// Copyright 2024 Smirnova Daria
#include <gtest/gtest.h>
#include <omp.h>

#include <chrono>
#include <complex>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/smirnova_d_complex_matrix_crs/include/ops_omp.hpp"

using namespace smirnova_omp;

crs_matrix generate_random_matrix(int n, int m, double proba, int seed) {
  std::mt19937 gen;
  gen.seed(seed);
  std::uniform_real_distribution<double> random(-2.0, 2.0);
  std::bernoulli_distribution bernoulli(proba);

  crs_matrix result;
  result.n_rows = n;
  result.n_cols = m;
  result.pointer.assign(result.n_rows + 1, 0);
  std::vector<std::vector<std::pair<int, std::complex<double>>>> temp(result.n_rows);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (bernoulli(gen)) {
        std::complex<double> val(random(gen), random(gen));
        temp[i].emplace_back(j, val);
      }
    }
  }
  for (int i = 0; i < result.n_rows; i++) {
    result.pointer[i + 1] = result.pointer[i];
    for (auto &j : temp[i]) {
      result.col_indexes.push_back(j.first);
      result.non_zero_values.push_back(j.second);
      result.pointer[i + 1]++;
    }
  }

  return result;
}

TEST(smirnova_d_complex_matrix_crs_omp, test_pipeline_run) {
  crs_matrix A = generate_random_matrix(100, 100, 0.6, 1993);
  crs_matrix B = generate_random_matrix(100, 100, 0.6, 4325);
  crs_matrix Result;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Result));

  // Create Task
  auto taskOmp = std::make_shared<TestComplexMatrixCrsPar>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 50;
  const auto t0 = omp_get_wtime();
  perfAttr->current_timer = [&] { return omp_get_wtime() - t0; };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskOmp);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(smirnova_d_complex_matrix_crs_omp, test_task_run) {
  crs_matrix A = generate_random_matrix(100, 100, 0.6, 1993);
  crs_matrix B = generate_random_matrix(100, 100, 0.6, 4325);
  crs_matrix Result;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Result));

  // Create Task
  auto taskOmp = std::make_shared<TestComplexMatrixCrsPar>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 50;
  const auto t0 = omp_get_wtime();
  perfAttr->current_timer = [&] { return omp_get_wtime() - t0; };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskOmp);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
