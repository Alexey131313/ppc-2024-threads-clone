// Copyright 2024 Kulagin Aleksandr
#include <gtest/gtest.h>
#include <oneapi/tbb.h>

#include "core/perf/include/perf.hpp"
#include "tbb/kulagin_a_gauss_filter_vert/include/ops_tbb.hpp"

TEST(kulagin_a_gauss_filter_vert_seq, test_pipeline_run) {
  // Create data
  size_t w = 3500;
  size_t h = 3500;
  float sigma = 2.0f;
  std::vector<uint32_t> img = kulagin_a_gauss::generator1(w, h);
  std::vector<float> kernel = kulagin_a_gauss::generate_kernel(sigma);
  std::vector<uint32_t> out(w * h);
  std::vector<uint32_t> res(w * h);
  kulagin_a_gauss::apply_filter(w, h, img.data(), kernel.data(), res.data());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(kernel.data()));
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(w);
  taskDataSeq->outputs_count.emplace_back(h);

  // Create Task
  auto myTask = std::make_shared<FilterGaussVerticalTaskTBBKulagin>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(myTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < w * h; i++) {
    ASSERT_EQ(res[i], out[i]);
  }
}

TEST(kulagin_a_gauss_filter_vert_seq, test_task_run) {
  // Create data
  size_t w = 3500;
  size_t h = 3500;
  float sigma = 2.0f;
  std::vector<uint32_t> img = kulagin_a_gauss::generator1(w, h);
  std::vector<float> kernel = kulagin_a_gauss::generate_kernel(sigma);
  std::vector<uint32_t> out(w * h);
  std::vector<uint32_t> res(w * h);
  kulagin_a_gauss::apply_filter(w, h, img.data(), kernel.data(), res.data());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(kernel.data()));
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(w);
  taskDataSeq->outputs_count.emplace_back(h);

  // Create Task
  auto myTask = std::make_shared<FilterGaussVerticalTaskTBBKulagin>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(myTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < w * h; i++) {
    ASSERT_EQ(res[i], out[i]);
  }
}
