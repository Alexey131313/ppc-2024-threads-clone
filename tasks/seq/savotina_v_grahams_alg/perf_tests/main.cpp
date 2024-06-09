// Copyright 2024 Savotina Valeria
#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/savotina_v_grahams_alg/include/ops_seq.hpp"

TEST(savotina_v_grahams_alg_seq, test_pipeline_run) {
  double min = 0;
  double max = 100000;
  size_t size = 300000;  // size of vector of random points

  // Create data
  std::vector<SavotinaPoint> points = {SavotinaPoint(max, min), SavotinaPoint(min, max), SavotinaPoint(max, max),
                                       SavotinaPoint(min, min)};
  std::vector<SavotinaPoint> randomPoints = SavotinaRandomPoints(min, max, size);
  points.insert(points.end(), randomPoints.begin(), randomPoints.end());
  std::vector<SavotinaPoint> res = {SavotinaPoint(min, min), SavotinaPoint(max, min), SavotinaPoint(max, max),
                                    SavotinaPoint(min, max)};
  std::vector<SavotinaPoint> mch(res.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> dataGrahamsAlgSeq = std::make_shared<ppc::core::TaskData>();
  dataGrahamsAlgSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  dataGrahamsAlgSeq->inputs_count.emplace_back(points.size());
  dataGrahamsAlgSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(mch.data()));
  dataGrahamsAlgSeq->outputs_count.emplace_back(mch.size());

  // Create Task
  auto testGrahamSequential = std::make_shared<SavotinaGrahamsAlgorithmSequential>(dataGrahamsAlgSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testGrahamSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(mch[i], res[i]);
  }
}

TEST(savotina_v_grahams_alg_seq, test_task_run) {
  double min = 0;
  double max = 100000;
  size_t size = 300000;  // size of vector of random points

  // Create data
  std::vector<SavotinaPoint> points = {SavotinaPoint(max, min), SavotinaPoint(min, max), SavotinaPoint(max, max),
                                       SavotinaPoint(min, min)};
  std::vector<SavotinaPoint> randomPoints = SavotinaRandomPoints(min, max, size);
  points.insert(points.end(), randomPoints.begin(), randomPoints.end());
  std::vector<SavotinaPoint> res = {SavotinaPoint(min, min), SavotinaPoint(max, min), SavotinaPoint(max, max),
                                    SavotinaPoint(min, max)};
  std::vector<SavotinaPoint> mch(res.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> dataGrahamsAlgSeq = std::make_shared<ppc::core::TaskData>();
  dataGrahamsAlgSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  dataGrahamsAlgSeq->inputs_count.emplace_back(points.size());
  dataGrahamsAlgSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(mch.data()));
  dataGrahamsAlgSeq->outputs_count.emplace_back(mch.size());

  // Create Task
  auto testGrahamSequential = std::make_shared<SavotinaGrahamsAlgorithmSequential>(dataGrahamsAlgSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testGrahamSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(mch[i], res[i]);
  }
}
