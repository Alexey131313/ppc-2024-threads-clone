// Copyright 2024 Korablev Nikita
#include <gtest/gtest.h>
#include <oneapi/tbb.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "tbb/korablev_n_montecarlo/include/ops_tbb.hpp"
#define ESTIMATE 0.2

TEST(TBB_Korablev_Nikita_perf_test, test_pipeline_run) {
  double res = 8;
  func f = flin;

  // Create data
  std::vector<double> in1 = {0, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataTBB->inputs_count.emplace_back(in1.size());
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataTBB->inputs_count.emplace_back(in2.size());

  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataTBB->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskTBB = std::make_shared<KorablevMonteCarloTBB>(taskDataTBB, f);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskTBB);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_LT(std::abs(res - out[0]), ESTIMATE);
}

TEST(TBB_Korablev_Nikita_perf_test, test_task_run) {
  double res = 8;
  func f = flin;

  // Create data
  std::vector<double> in1 = {0, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataTBB->inputs_count.emplace_back(in1.size());
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataTBB->inputs_count.emplace_back(in2.size());

  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataTBB->outputs_count.emplace_back(out.size());
  // Create Task
  auto testTaskTBB = std::make_shared<KorablevMonteCarloTBB>(taskDataTBB, f);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskTBB);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_LT(std::abs(res - out[0]), ESTIMATE);
}