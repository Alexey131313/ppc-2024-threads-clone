// Copyright 2024 Kasimtcev Roman
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kasimtcev_r_monte_carlo/include/ops_seq.hpp"
#define ESTIMATE 0.2

TEST(sequential_kasimtcev_r_perf_test, test_add) {
  double res = 1.5;
  func f = fxyy;

  std::vector<double> a1 = {0, 2};
  std::vector<double> a2 = {0, 2};
  std::vector<double> out(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataSeq->inputs_count.emplace_back(a1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataSeq->inputs_count.emplace_back(a2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(a2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<MonteCarloSequentialKasimtcev>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_LT(std::abs(res - out[0]), ESTIMATE);
}

TEST(sequential_kasimtcev_r_perf_test, test_miltmilt) {
  double res = 5.67369;
  func f = fsinxsiny;

  std::vector<double> a1 = {0, 2};
  std::vector<double> a2 = {0, 2};
  std::vector<double> out(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataSeq->inputs_count.emplace_back(a1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataSeq->inputs_count.emplace_back(a2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(a2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<MonteCarloSequentialKasimtcev>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_LT(std::abs(res - out[0]), ESTIMATE);
}
