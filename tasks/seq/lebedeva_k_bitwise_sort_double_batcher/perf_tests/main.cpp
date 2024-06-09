// Copyright Lebedeva_Ksenia 2024
#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/lebedeva_k_bitwise_sort_double_batcher/include/ops_seq.hpp"

TEST(lebedeva_k_bitwise_sort_double_batcher, test_pipeline_run) {
  // Create data
  std::vector<double> array = randomVector(50000, -1000000, 1000000);
  std::vector<double> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> RSDBDataSeq = std::make_shared<ppc::core::TaskData>();
  RSDBDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  RSDBDataSeq->inputs_count.emplace_back(array.size());
  RSDBDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  RSDBDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  auto testRSDBSequential = std::make_shared<RadixSortDoubleBatcherSequential>(RSDBDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testRSDBSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(lebedeva_k_bitwise_sort_double_batcher, test_task_run) {
  // Create data
  std::vector<double> array = randomVector(50000, -1000000, 1000000);
  std::vector<double> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> RSDBDataSeq = std::make_shared<ppc::core::TaskData>();
  RSDBDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  RSDBDataSeq->inputs_count.emplace_back(array.size());
  RSDBDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  RSDBDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  auto testRSDBSequential = std::make_shared<RadixSortDoubleBatcherSequential>(RSDBDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testRSDBSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}
