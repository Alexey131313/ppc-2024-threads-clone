// Copyright 2024 Kokin Ivan
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kokin_marking_component_bin_image/include/ops_seq.hpp"

TEST(kokin_marking_component_bin_image_test, test_run) {
  uint32_t ht = 1200;
  uint32_t wh = 1200;
  std::vector<uint32_t> size = {ht, wh};
  std::vector<uint8_t> in(ht * wh, 0);
  std::vector<uint8_t> out(ht * wh, 0);
  std::vector<uint8_t> cmpt(ht * wh, 1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(size.data()));
  taskDataSeq->inputs_count.emplace_back(size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSeq = std::make_shared<imageMarkingSeq>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 100;
  const auto tm = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(time_point - tm).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSeq);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(out, cmpt);
}

TEST(kokin_marking_component_bin_image_test, test_task_run) {
  uint32_t ht = 1200;
  uint32_t wh = 1200;
  std::vector<uint32_t> size = {ht, wh};
  std::vector<uint8_t> in(ht * wh, 0);
  std::vector<uint8_t> out(ht * wh, 0);
  std::vector<uint8_t> cmpt(ht * wh, 1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(size.data()));
  taskDataSeq->inputs_count.emplace_back(size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSeq = std::make_shared<imageMarkingSeq>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto tm = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(time_point - tm).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSeq);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(out, cmpt);
}
