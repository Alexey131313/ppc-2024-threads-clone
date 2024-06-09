// Copyright 2024 Savchuk Anton
#include <gtest/gtest.h>
#include <omp.h>

#include "core/perf/include/perf.hpp"
#include "omp/savchuk_a_crs_matmult_omp/include/crs_matmult_omp.hpp"

using namespace SavchukOMP;

TEST(savchuk_a_crs_matmult_omp, test_pipeline_run) {
  // Create data
  size_t p = 501;
  size_t q = 500;
  size_t r = 501;
  std::vector<Complex> lhs_in(p * q, Complex(0, 0));
  for (size_t i = 0; i < p; ++i) {
    if (i % 4 == 0)
      for (size_t j = 0; j < q; ++j) {
        lhs_in[i * q + j] = Complex(1.0, 1.0);
      }
  }
  std::vector<Complex> rhs_in(q * r, Complex(0, 0));
  for (size_t i = 0; i < q; ++i) {
    for (size_t j = 0; j < r; ++j) {
      if (j % 4 == 0) rhs_in[i * r + j] = Complex(1.0, -1.0);
    }
  }
  std::vector<Complex> out(p * r);

  std::vector<Complex> res(p * r, Complex(0, 0));
  for (size_t i = 0; i < p; ++i) {
    for (size_t j = 0; j < r; ++j) {
      if (i % 4 == 0 && j % 4 == 0) {
        res[i * r + j] = Complex(2 * q, 0);
      }
    }
  }

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
  auto testTaskOMP = std::make_shared<SavchukCRSMatMultOMPParallel>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOMP);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(res[i], out[i]);
  }
}

TEST(savchuk_a_crs_matmult_omp, test_task_run) {
  size_t p = 501;
  size_t q = 500;
  size_t r = 501;
  std::vector<Complex> lhs_in(p * q, Complex(0, 0));
  for (size_t i = 0; i < p; ++i) {
    if (i % 4 == 0)
      for (size_t j = 0; j < q; ++j) {
        lhs_in[i * q + j] = Complex(1.0, 1.0);
      }
  }
  std::vector<Complex> rhs_in(q * r, Complex(0, 0));
  for (size_t i = 0; i < q; ++i) {
    for (size_t j = 0; j < r; ++j) {
      if (j % 4 == 0) rhs_in[i * r + j] = Complex(1.0, -1.0);
    }
  }
  std::vector<Complex> out(p * r);

  std::vector<Complex> res(p * r, Complex(0, 0));
  for (size_t i = 0; i < p; ++i) {
    for (size_t j = 0; j < r; ++j) {
      if (i % 4 == 0 && j % 4 == 0) {
        res[i * r + j] = Complex(2 * q, 0);
      }
    }
  }

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
  auto testTaskOMP = std::make_shared<SavchukCRSMatMultOMPParallel>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOMP);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(res[i], out[i]);
  }
}