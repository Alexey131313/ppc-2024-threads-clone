// Copyright 2024 Kasimtcev Roman
#include <gtest/gtest.h>
#include <omp.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/kasimtcev_r_montecarlo/include/ops_omp.hpp"
#define ESTIMATE 0.2

TEST(omp_kasimtcev_r_perf_test, test_pipeline_run) {
  double res = 8;
  kasimtcev_func f = kasimtcev_flinear;

  std::vector<double> a1 = {0, 2};
  std::vector<double> a2 = {0, 2};
  std::vector<double> out(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataSeq->inputs_count.emplace_back(a1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataSeq->inputs_count.emplace_back(a2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskOMP = std::make_shared<KasimtcevOMPMonteCarlo>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOMP);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_LT(std::abs(res - out[0]), ESTIMATE);
}

TEST(omp_kasimtcev_r_perf_test, test_task_run) {
  double res = 8;
  kasimtcev_func f = kasimtcev_flinear;

  std::vector<double> a1 = {0, 2};
  std::vector<double> a2 = {0, 2};
  std::vector<double> out(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataSeq->inputs_count.emplace_back(a1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataSeq->inputs_count.emplace_back(a2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  auto testTaskOMP = std::make_shared<KasimtcevOMPMonteCarlo>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOMP);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_LT(std::abs(res - out[0]), ESTIMATE);
}
