// Copyright 2024 Polozov Vladislav
#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/polozov_v_sort_hoar_batcher/include/ops_seq.hpp"

TEST(polozov_v_sort_hoar_batcher_seq, test_pipeline_run) {
  // Create data
  constexpr int size = (1 << 20);
  std::vector<int> array = generate_data(size, -1000000, 1000000);
  std::vector<int> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> DataSeq = std::make_shared<ppc::core::TaskData>();
  DataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  DataSeq->inputs_count.emplace_back(array.size());
  DataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  DataSeq->outputs_count.emplace_back(out.size());

  std::vector<int> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  auto myTask = std::make_shared<SortHoarWithMergeBatcher>(DataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(myTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(polozov_v_sort_hoar_batcher_seq, test_task_run) {
  // Create data
  constexpr int size = (1 << 20);
  std::vector<int> array = generate_data(size, -1000000, 1000000);
  std::vector<int> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> DataSeq = std::make_shared<ppc::core::TaskData>();
  DataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  DataSeq->inputs_count.emplace_back(array.size());
  DataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  DataSeq->outputs_count.emplace_back(out.size());

  std::vector<int> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  auto myTask = std::make_shared<SortHoarWithMergeBatcher>(DataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(myTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}