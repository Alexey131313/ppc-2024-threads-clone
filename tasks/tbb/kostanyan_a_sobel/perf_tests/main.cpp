// Copyright 2024 Kostanyan Arsen
#include <gtest/gtest.h>
#include <oneapi/tbb.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "tbb/kostanyan_a_sobel/include/ops_tbb.hpp"

TEST(tbb_kostanyan_sobel_perf_test, test_pipeline_run) {
  const int n = 3000;
  const int m = 3000;
  const uint8_t min = 0;
  const uint8_t max = 255;

  // Create data
  std::vector<int> in(2);
  in[0] = n;
  in[1] = m;
  std::vector<uint8_t> pict = kostanyan_tbb_sobel::kostanyan_getRandomPicture(n, m, min, max);
  std::vector<uint8_t> out(n * m, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(pict.data());
  taskDataSeq->inputs_count.emplace_back(pict.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskTBB = std::make_shared<kostanyan_tbb_sobel::Kostanyan_EdgeDetectionTBBParallel>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskTBB);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (int i = 1; i < n - 1; i++) {
    for (int j = 1; j < m - 1; j++) {
      int gx = -pict[(i - 1) * m + j - 1] + pict[(i - 1) * m + j + 1] - 2 * pict[i * m + j - 1] +
               2 * pict[i * m + j + 1] - pict[(i + 1) * m + j - 1] + pict[(i + 1) * m + j + 1];
      int gy = pict[(i - 1) * m + j - 1] + 2 * pict[(i - 1) * m + j] + pict[(i - 1) * m + j + 1] -
               pict[(i + 1) * m + j - 1] - 2 * pict[(i + 1) * m + j] - pict[(i + 1) * m + j + 1];
      uint8_t expected = sqrt(gx * gx + gy * gy);
      ASSERT_EQ(expected, out[i * m + j]);
    }
  }
}

TEST(tbb_kostanyan_sobel_perf_test, test_task_run) {
  const int n = 3000;
  const int m = 3000;
  const uint8_t min = 0;
  const uint8_t max = 255;

  // Create data
  std::vector<int> in(2);
  in[0] = n;
  in[1] = m;
  std::vector<uint8_t> pict = kostanyan_tbb_sobel::kostanyan_getRandomPicture(n, m, min, max);
  std::vector<uint8_t> out(n * m, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(pict.data());
  taskDataSeq->inputs_count.emplace_back(pict.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskTBB = std::make_shared<kostanyan_tbb_sobel::Kostanyan_EdgeDetectionTBBParallel>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskTBB);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (int i = 1; i < n - 1; i++) {
    for (int j = 1; j < m - 1; j++) {
      int gx = -pict[(i - 1) * m + j - 1] + pict[(i - 1) * m + j + 1] - 2 * pict[i * m + j - 1] +
               2 * pict[i * m + j + 1] - pict[(i + 1) * m + j - 1] + pict[(i + 1) * m + j + 1];
      int gy = pict[(i - 1) * m + j - 1] + 2 * pict[(i - 1) * m + j] + pict[(i - 1) * m + j + 1] -
               pict[(i + 1) * m + j - 1] - 2 * pict[(i + 1) * m + j] - pict[(i + 1) * m + j + 1];
      uint8_t expected = sqrt(gx * gx + gy * gy);
      ASSERT_EQ(expected, out[i * m + j]);
    }
  }
}