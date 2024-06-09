// Copyright 2024 Nogin Denis
#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "omp/nogin_d_radix_double_batcher/include/ops_omp.hpp"

TEST(nogin_d_radix_double_batcher_omp, test_pipeline_run) {
  // Create data
  std::vector<double> array = NoginDenisOmp::randomVector(1000000, -1000000, 1000000);
  std::vector<double> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> RSDBDataOmp = std::make_shared<ppc::core::TaskData>();
  RSDBDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  RSDBDataOmp->inputs_count.emplace_back(array.size());
  RSDBDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  RSDBDataOmp->outputs_count.emplace_back(out.size());

  std::vector<double> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  auto testRSDBompParallel = std::make_shared<NoginDenisOmp::RadixSortDoubleBatcherOmpParallel>(RSDBDataOmp);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testRSDBompParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(nogin_d_radix_double_batcher_omp, test_task_run) {
  // Create data
  std::vector<double> array = NoginDenisOmp::randomVector(1000000, -1000000, 1000000);
  std::vector<double> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> RSDBDataOmp = std::make_shared<ppc::core::TaskData>();
  RSDBDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  RSDBDataOmp->inputs_count.emplace_back(array.size());
  RSDBDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  RSDBDataOmp->outputs_count.emplace_back(out.size());

  std::vector<double> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  auto testRSDBompParallel = std::make_shared<NoginDenisOmp::RadixSortDoubleBatcherOmpParallel>(RSDBDataOmp);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testRSDBompParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}
