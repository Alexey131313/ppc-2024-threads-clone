// Copyright 2024 Shokurov Daniil
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/shokurov_d_convex_hull/include/ops_omp.hpp"

TEST(shokurov_d_convex_hull_omp, test_pipeline_run) {
  // Create data
  std::vector<pair<double, double>> in;
  std::vector<pair<double, double>> ans;
  ans.emplace_back(0, 0);
  ans.emplace_back(1, 0);
  ans.emplace_back(5, 5);
  ans.emplace_back(0, 8);
  ans.emplace_back(-3, 1);
  size_t co = 5;
  size_t n = 5000000;
  for (size_t i = 0; i < co; ++i) {
    in.emplace_back(ans[i]);
  }

  std::mt19937 gen(1.0);
  std::uniform_real_distribution<> dis(0.1, 0.9);

  for (size_t i = 0; i < n; ++i) {
    size_t x = i % co;
    size_t y = (i + 1) % co;
    double a = dis(gen);
    pair<double, double> np;
    np.first = ans[x].first * a + (1.0 - a) * ans[y].first;
    np.second = ans[x].second * a + (1.0 - a) * ans[y].second;
    in.emplace_back(np);
  }
  std::vector<pair<double, double>> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test = std::make_shared<ConvexHullSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Check answer
  size_t k = taskDataSeq->outputs_count[0];
  ASSERT_EQ(ans.size(), k);

  auto *_out = reinterpret_cast<pair<double, double> *>(taskDataSeq->outputs[0]);

  std::sort(_out, _out + k);
  std::sort(ans.begin(), ans.end());

  for (size_t i = 0; i < k; ++i) {
    ASSERT_DOUBLE_EQ(_out[i].first, ans[i].first);
    ASSERT_DOUBLE_EQ(_out[i].second, ans[i].second);
  }
}

TEST(shokurov_d_convex_hull_omp, test_task_run) {
  // Create data
  std::vector<pair<double, double>> in;
  std::vector<pair<double, double>> ans;
  ans.emplace_back(0, 0);
  ans.emplace_back(1, 0);
  ans.emplace_back(5, 5);
  ans.emplace_back(0, 8);
  ans.emplace_back(-3, 1);
  size_t co = 5;
  size_t n = 5000000;
  for (size_t i = 0; i < co; ++i) {
    in.emplace_back(ans[i]);
  }

  std::mt19937 gen(1.0);
  std::uniform_real_distribution<> dis(0.1, 0.9);

  for (size_t i = 0; i < n; ++i) {
    size_t x = i % co;
    size_t y = (i + 1) % co;
    double a = dis(gen);
    pair<double, double> np;
    np.first = ans[x].first * a + (1.0 - a) * ans[y].first;
    np.second = ans[x].second * a + (1.0 - a) * ans[y].second;
    in.emplace_back(np);
  }
  std::vector<pair<double, double>> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test = std::make_shared<ConvexHullSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Check answer
  size_t k = taskDataSeq->outputs_count[0];
  ASSERT_EQ(ans.size(), k);

  auto *_out = reinterpret_cast<pair<double, double> *>(taskDataSeq->outputs[0]);

  std::sort(_out, _out + k);
  std::sort(ans.begin(), ans.end());

  for (size_t i = 0; i < k; ++i) {
    ASSERT_DOUBLE_EQ(_out[i].first, ans[i].first);
    ASSERT_DOUBLE_EQ(_out[i].second, ans[i].second);
  }
}
