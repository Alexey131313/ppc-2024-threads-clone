// Copyright 2024 Kurdina Julia
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kurdina_j_linear_filter/include/ops_seq.hpp"

TEST(kurdina_j_linear_filter_seq, test_pipeline_run) {
  int n = 700;
  int m = 700;
  std::vector<uint8_t> gauss_kernel = {1, 1, 1, 1, 9, 1, 1, 1, 1};

  // Create data
  std::vector<uint8_t> in(n * m, 255);
  std::vector<int> size_m = {n, m};
  std::vector<uint8_t> out(n * m);
  std::vector<uint8_t> result(n * m, 255);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(size_m.data()));
  taskDataSeq->inputs_count.emplace_back(size_m.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(gauss_kernel.data()));
  taskDataSeq->inputs_count.emplace_back(gauss_kernel.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<HorizontalSplitSequential>(taskDataSeq);

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
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(kurdina_j_linear_filter_seq, test_task_run) {
  int n = 700;
  int m = 700;
  std::vector<uint8_t> gauss_kernel = {1, 1, 1, 1, 9, 1, 1, 1, 1};

  // Create data
  std::vector<uint8_t> in(n * m, 255);
  std::vector<int> size_m = {n, m};
  std::vector<uint8_t> out(n * m);
  std::vector<uint8_t> result(n * m, 255);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(size_m.data()));
  taskDataSeq->inputs_count.emplace_back(size_m.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(gauss_kernel.data()));
  taskDataSeq->inputs_count.emplace_back(gauss_kernel.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<HorizontalSplitSequential>(taskDataSeq);

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
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(result[i], out[i]);
  }
}
