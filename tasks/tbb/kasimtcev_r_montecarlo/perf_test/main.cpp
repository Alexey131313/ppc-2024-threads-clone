// Copyright 2024 Kasimtcev Roman
#include <gtest/gtest.h>
#include <oneapi/tbb.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "tbb/kasimtcev_r_montecarlo/include/ops_tbb.hpp"
#define ESTIMATE 0.2

TEST(TBB_kasimtcev_r_perftest, tested_pipeline) {
  double res = 8;
  func f = flinear;

  std::vector<double> a1 = {0, 2};
  std::vector<double> a2 = {0, 2};
  std::vector<double> out(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataTBB->inputs_count.emplace_back(a1.size());
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataTBB->inputs_count.emplace_back(a2.size());

  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataTBB->outputs_count.emplace_back(out.size());

  auto testTaskTBB = std::make_shared<KasimtcevMonteCarloTBB>(taskDataTBB, f);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskTBB);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_LT(std::abs(res - out[0]), ESTIMATE);
}

TEST(TBB_kasimtcev_r_perftest, tested_task) {
  double res = 8;
  func f = flinear;

  std::vector<double> a1 = {0, 2};
  std::vector<double> a2 = {0, 2};
  std::vector<double> out(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataTBB->inputs_count.emplace_back(a1.size());
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataTBB->inputs_count.emplace_back(a2.size());

  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataTBB->outputs_count.emplace_back(out.size());

  auto testTaskTBB = std::make_shared<KasimtcevMonteCarloTBB>(taskDataTBB, f);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskTBB);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_LT(std::abs(res - out[0]), ESTIMATE);
}
