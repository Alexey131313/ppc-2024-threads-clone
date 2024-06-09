// Copyright 2024 Zorin Oleg
#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "tbb/tbb.h"
#include "tbb/zorin_o_crs_matmult/include/crs_matmult_tbb.hpp"

TEST(Zorin_O_CRS_MatMult_TBB, test_pipeline_run) {
  // Create data
  int p = 901;
  int q = 900;
  int r = 901;
  std::vector<double> lhs_in(p * q);
  for (int i = 0; i < p; ++i) {
    if (i % 2 == 0)
      for (int j = 0; j < q; ++j) {
        lhs_in[i * q + j] = 1.0;
      }
  }
  std::vector<double> rhs_in(q * r);
  for (int i = 0; i < q; ++i) {
    for (int j = 0; j < r; ++j) {
      if (j % 2 == 0) rhs_in[i * r + j] = 1.0;
    }
  }
  std::vector<double> out(p * r);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(lhs_in.data()));
  taskDataTBB->inputs_count.emplace_back(p);
  taskDataTBB->inputs_count.emplace_back(q);
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs_in.data()));
  taskDataTBB->inputs_count.emplace_back(q);
  taskDataTBB->inputs_count.emplace_back(r);
  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataTBB->outputs_count.emplace_back(p);
  taskDataTBB->outputs_count.emplace_back(r);

  // Create Task
  auto testTaskTBB = std::make_shared<CRSMatMult>(taskDataTBB);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskTBB);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < r; ++j) {
      if (i % 2 == 0 && j % 2 == 0)
        EXPECT_DOUBLE_EQ(out[i * r + j], q);
      else
        EXPECT_DOUBLE_EQ(out[i * r + j], 0.0);
    }
  }
}

TEST(Zorin_O_CRS_MatMult_TBB, test_task_run) {
  // Create data
  int p = 901;
  int q = 900;
  int r = 901;
  std::vector<double> lhs_in(p * q);
  for (int i = 0; i < p; ++i) {
    if (i % 2 == 0)
      for (int j = 0; j < q; ++j) {
        lhs_in[i * q + j] = 1.0;
      }
  }
  std::vector<double> rhs_in(q * r);
  for (int i = 0; i < q; ++i) {
    for (int j = 0; j < r; ++j) {
      if (j % 2 == 0) rhs_in[i * r + j] = 1.0;
    }
  }
  std::vector<double> out(p * r);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(lhs_in.data()));
  taskDataTBB->inputs_count.emplace_back(p);
  taskDataTBB->inputs_count.emplace_back(q);
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(rhs_in.data()));
  taskDataTBB->inputs_count.emplace_back(q);
  taskDataTBB->inputs_count.emplace_back(r);
  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataTBB->outputs_count.emplace_back(p);
  taskDataTBB->outputs_count.emplace_back(r);

  // Create Task
  auto testTaskTBB = std::make_shared<CRSMatMult>(taskDataTBB);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskTBB);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < r; ++j) {
      if (i % 2 == 0 && j % 2 == 0)
        EXPECT_DOUBLE_EQ(out[i * r + j], q);
      else
        EXPECT_DOUBLE_EQ(out[i * r + j], 0.0);
    }
  }
}
