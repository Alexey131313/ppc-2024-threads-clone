// Copyright 2024 Mitin Roman
#include <gtest/gtest.h>
#include <omp.h>

#include <limits>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/mitin_r_double_radix_sort/include/ops_omp.hpp"

namespace {

constexpr size_t input_size = 5e+6;

}  // namespace

double timer() { return omp_get_wtime(); }

TEST(mitin_r_double_radix_sort_seq, test_pipeline_run) {
  // Create data

  std::mt19937 gen(1.0);
  std::uniform_int_distribution<int64_t> dis(0.0, std::numeric_limits<int64_t>::max());
  std::vector<double> in(input_size);
  std::vector<double> expected(input_size);
  for (size_t i = 0; i < input_size; i++) {
    uint64_t rand_val = dis(gen);
    memcpy(&expected[i], &rand_val, sizeof(rand_val));
    in[i] = expected[i];
  }

  std::vector<double *> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  auto testTaskSequential = std::make_shared<SortRadixDoubleTaskOMP>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;
  perfAttr->current_timer = &timer;

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::sort(expected.begin(), expected.end());
  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[0][i], expected[i]);
  }
}

TEST(mitin_r_double_radix_sort_seq, test_task_run) {
  // Create data

  std::mt19937 gen(1.0);
  std::uniform_int_distribution<int64_t> dis(0.0, std::numeric_limits<int64_t>::max());
  std::vector<double> in(input_size);
  std::vector<double> expected(input_size);
  for (size_t i = 0; i < input_size; i++) {
    uint64_t rand_val = dis(gen);
    memcpy(&expected[i], &rand_val, sizeof(rand_val));
    in[i] = expected[i];
  }

  std::vector<double *> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  auto testTaskSequential = std::make_shared<SortRadixDoubleTaskOMP>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;
  perfAttr->current_timer = &timer;

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::sort(expected.begin(), expected.end());
  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[0][i], expected[i]);
  }
}
