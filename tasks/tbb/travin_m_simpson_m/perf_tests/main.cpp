// Copyright 2024 Travin Maksim
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "tbb/travin_m_simpson_m/include/ops_tbb.hpp"
using namespace travin_tbb;

TEST(travin_m_simpson_tbb, test_pipeline_run) {
  int a = 6;
  int b = 10;
  int c = 6;
  int d = 10;
  int n = 2000;
  double res = 128;
  function func = LinFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> out{res};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataTbb->inputs_count.emplace_back(in.size());
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataTbb->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskTBB = std::make_shared<SimpsonTaskTbb>(taskDataTbb, func);

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
  ASSERT_NEAR(res, out[0], 0.001);
}

TEST(travin_m_simpson_tbb, test_task_run) {
  int a = 6;
  int b = 10;
  int c = 6;
  int d = 10;
  int n = 2000;
  double res = 128;
  function func = LinFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> out{res};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataTbb->inputs_count.emplace_back(in.size());
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataTbb->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskTBB = std::make_shared<SimpsonTaskTbb>(taskDataTbb, func);

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
  ASSERT_NEAR(res, out[0], 0.001);
}