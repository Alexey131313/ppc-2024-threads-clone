// Copyright 2024 Konovalov Igor
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/konovalov_i_radix_sort_doubles_s_m/include/ops_seq.hpp"

TEST(konovalov_i_double_radix_sort_seq, test_pipeline_run) {
  const size_t size = 1e+5;
  std::mt19937 gen(1.0);
  std::uniform_real_distribution<> dis(0.0, 1e+5);

  // Create data
  std::vector<double> in(size);
  std::vector<double> out(size);
  for (size_t i = 0; i < size; i++) {
    double a = dis(gen);
    in[i] = a;
    out[i] = a;
  }
  std::sort(out.begin(), out.end());
  std::vector<double *> res(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  auto testTaskSequential = std::make_shared<RadixSortSequentialTask>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < size; i++) {
    if (out[i] != res[0][i]) {
      FAIL();
    }
  }
  SUCCEED();
}

TEST(konovalov_i_double_radix_sort_seq, test_task_run) {
  const size_t size = 1e+5;
  std::mt19937 gen(1.0);
  std::uniform_real_distribution<> dis(0.0, 1e+5);

  // Create data
  std::vector<double> in(size);
  std::vector<double> out(size);
  for (size_t i = 0; i < size; i++) {
    double a = dis(gen);
    in[i] = a;
    out[i] = a;
  }
  std::sort(out.begin(), out.end());
  std::vector<double *> res(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  auto testTaskSequential = std::make_shared<RadixSortSequentialTask>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < size; i++) {
    if (out[i] != res[0][i]) {
      FAIL();
    }
  }
  SUCCEED();
}
