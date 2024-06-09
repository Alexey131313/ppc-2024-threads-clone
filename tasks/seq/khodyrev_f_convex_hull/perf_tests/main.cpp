// Copyright 2024 Khodyrev Fedor
#include <gtest/gtest.h>

#include <stack>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/khodyrev_f_convex_hull/include/ops_seq.hpp"

TEST(khodyrev_f_convex_null_seq, test_pipeline_run) {
  int height_in = 13000;
  int width_in = 1000;
  int height_out = 13000;
  int width_out = 1000;
  std::vector<int> in(13000000, 0);
  in[0] = 1;
  in[1] = 1;
  std::vector<int> out(height_out * width_out, 0);

  std::vector<int> true_result(13000000, 0);
  true_result[0] = 1;
  true_result[1] = 1;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(height_in);
  taskDataSeq->inputs_count.emplace_back(width_in);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(height_out);
  taskDataSeq->outputs_count.emplace_back(width_out);

  auto testTaskSequential = std::make_shared<KhodyrevTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (std::vector<int>::size_type i = 0; i < in.size(); i++) {
    ASSERT_EQ(out[i], true_result[i]);
  }
}

TEST(khodyrev_f_convex_null_seq, test_task_run) {
  int height_in = 13000;
  int width_in = 1000;
  int height_out = 13000;
  int width_out = 1000;
  std::vector<int> in(13000000, 0);
  in[0] = 1;
  in[1] = 1;
  std::vector<int> out(height_out * width_out, 0);

  std::vector<int> true_result(13000000, 0);
  true_result[0] = 1;
  true_result[1] = 1;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(height_in);
  taskDataSeq->inputs_count.emplace_back(width_in);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(height_out);
  taskDataSeq->outputs_count.emplace_back(width_out);

  auto testTaskSequential = std::make_shared<KhodyrevTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (std::vector<int>::size_type i = 0; i < in.size(); i++) {
    ASSERT_EQ(out[i], true_result[i]);
  }
}
