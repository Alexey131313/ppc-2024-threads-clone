// Copyright 2024 Videneva Ekaterina
#include <gtest/gtest.h>
#include <omp.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/videneva_int_metod_pr_omp_/include/ops_omp.hpp"

TEST(videneva_int_metod_pr_omp_perf, pipRun) {
  std::vector<double> in{-15.0, 15.0, -20.0, 10.0, 4e3};
  std::vector<double> out(2);

  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  data_seq->inputs_count.emplace_back(in.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(out.size());

  auto videneva_opm_task = std::make_shared<Videneva_opm_task>(data_seq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(videneva_opm_task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_NEAR((double)63000, out[0], out[1]);
}

TEST(videneva_int_metod_pr_omp_perf, test_task_run) {
  std::vector<double> in{-15.0, 15.0, -20.0, 10.0, 4e3};
  std::vector<double> out(2);

  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  data_seq->inputs_count.emplace_back(in.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(out.size());

  auto videneva_opm_task = std::make_shared<Videneva_opm_task>(data_seq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(videneva_opm_task);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_NEAR((double)63000, out[0], out[1]);
}