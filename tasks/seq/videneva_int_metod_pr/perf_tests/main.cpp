// Copyright 2023 Videneva Ekaterina
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/videneva_int_metod_pr/include/ops_seq.hpp"

TEST(videneva_e_int_metod_pr_perf_tests, pipRun) {
  std::vector<double> in{-15.0, 15.0, -20.0, 10.0, 4e3};
  std::vector<double> out(2);

  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  data_seq->inputs_count.emplace_back(in.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(out.size());

  auto videneva_seq_task = std::make_shared<Videneva_seq_task>(data_seq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(videneva_seq_task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_NEAR((double)63000, out[0], out[1]);
}

TEST(videneva_e_int_metod_pr_perf_tests, tasksRun) {
  std::vector<double> in{-15.0, 15.0, -20.0, 10.0, 4e3};
  std::vector<double> out(2);

  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  data_seq->inputs_count.emplace_back(in.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(out.size());

  auto videneva_seq_task = std::make_shared<Videneva_seq_task>(data_seq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(videneva_seq_task);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_NEAR((double)63000, out[0], out[1]);
}