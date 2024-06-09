// Copyright 2023 Safronov Mikhail
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/safronov_m/include/ops_seq.hpp"

TEST(Safronov_mult_m_fox, test_pipeline_run) {
  size_t n = 300;
  double k = 50.0;
  std::vector<double> a(n * n);
  std::vector<double> b(n * n);
  std::vector<double> res(n * n);
  GetRandomValue(a.data(), a.size());
  identityMatrix(b.data(), n, k);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(a.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  auto safronovTaskSequential = std::make_shared<SafronovSeqFoxAlgTask>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(safronovTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < n * n; i++) {
    EXPECT_DOUBLE_EQ(res[i], k * a[i]);
  }
}

TEST(Safronov_mult_m_fox, test_task_run) {
  size_t n = 300;
  double k = 50.0;
  std::vector<double> a(n * n);
  std::vector<double> b(n * n);
  std::vector<double> res(n * n);
  GetRandomValue(a.data(), a.size());
  identityMatrix(b.data(), n, k);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(a.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  auto safronovTaskSequential = std::make_shared<SafronovSeqFoxAlgTask>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(safronovTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < n * n; i++) {
    EXPECT_DOUBLE_EQ(res[i], k * a[i]);
  }
}
