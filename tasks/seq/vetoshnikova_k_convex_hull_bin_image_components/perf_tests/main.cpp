// Copyright 2024 Vetoshnikova Ekaterina
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vetoshnikova_k_convex_hull_bin_image_components/include/ops_seq.hpp"

TEST(vetoshnikova_k_hull_bin_image_seq, test_pipeline_run) {
  int h = 200;
  int w = 200;
  // Create data
  std::vector<int> out(w * h);
  std::vector<uint8_t> in(h * w, 0);

  for (int i = 0; i < h * w / 2; i++) {
    in[i] = 1;
  }
  std::vector<int> hullTrue(4 * w + h - 3, 0);

  int i;
  for (i = 0; i < w * 2; i++) {
    if (i % 2 == 0)
      hullTrue[i] = 0;
    else
      hullTrue[i] = i / 2;
  }
  int k = 1;
  while (k < h / 2) {
    hullTrue[i] = k;
    hullTrue[i + 1] = w - 1;
    i += 2;
    k++;
  }
  k--;
  int m = w - 2;
  while (m >= 0) {
    hullTrue[i] = k;
    hullTrue[i + 1] = m;
    m--;
    i += 2;
  }
  hullTrue[i] = -1;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));

  // Create Task
  auto testTaskSequential = std::make_shared<ConstructingConvexHullSeq>(taskDataSeq);

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
  for (size_t j = 0; j < hullTrue.size(); j++) {
    ASSERT_EQ(hullTrue[j], out[j]);
  }
}

TEST(vetoshnikova_k_hull_bin_image_seq, test_task_run) {
  int h = 200;
  int w = 200;
  // Create data
  std::vector<int> out(w * h);
  std::vector<uint8_t> in(h * w, 0);

  for (int i = 0; i < h * w / 2; i++) {
    in[i] = 1;
  }
  std::vector<int> hullTrue(4 * w + h - 3, 0);

  int i;
  for (i = 0; i < w * 2; i++) {
    if (i % 2 == 0)
      hullTrue[i] = 0;
    else
      hullTrue[i] = i / 2;
  }
  int k = 1;
  while (k < h / 2) {
    hullTrue[i] = k;
    hullTrue[i + 1] = w - 1;
    i += 2;
    k++;
  }
  k--;
  int m = w - 2;
  while (m >= 0) {
    hullTrue[i] = k;
    hullTrue[i + 1] = m;
    m--;
    i += 2;
  }
  hullTrue[i] = -1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));

  // Create Task
  auto testTaskSequential = std::make_shared<ConstructingConvexHullSeq>(taskDataSeq);

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

  for (size_t j = 0; j < hullTrue.size(); j++) {
    ASSERT_EQ(hullTrue[j], out[j]);
  }
}
