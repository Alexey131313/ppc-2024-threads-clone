// Copyright 2024 Kuznetsov Artem
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "tbb/kuznetsov_a_cannon_matr_mult/include/ops_tbb.hpp"

TEST(Kuznetsov_a_cannon_matr_mult_tbb_perf_tests, test_900x900) {
  // Create data
  size_t size = 900;
  size_t block = 450;

  double minVal = -100.0;
  double maxVal = 100.0;

  auto inputMatrOne = KuznetsovArtyomTbb::getRandomSquareMatrix(size, minVal, maxVal);
  auto inputMatrTwo = KuznetsovArtyomTbb::getRandomSquareMatrix(size, minVal, maxVal);
  std::vector<double> outputMatr(size * size, 0.0);

  // Create TaskData
  auto taskDataTbb = std::make_shared<ppc::core::TaskData>();

  // Add matrOne
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrOne.data()));
  taskDataTbb->inputs_count.emplace_back(inputMatrOne.size());

  // Add matrTwo
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrTwo.data()));
  taskDataTbb->inputs_count.emplace_back(inputMatrTwo.size());

  // Add size
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));

  // Add block
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block));

  // Add out matr
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputMatr.data()));
  taskDataTbb->outputs_count.emplace_back(outputMatr.size());

  auto resSeq = KuznetsovArtyomTbb::CannonMatrixMultSeq(inputMatrOne, inputMatrTwo, size, block);

  // Create Task
  auto testTaskTbb = std::make_shared<KuznetsovArtyomTbb::KuznetsovCannonMatrMultTbb>(taskDataTbb);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskTbb);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  auto resSize = resSeq.size();

  for (size_t i = 0; i < resSize; ++i) {
    ASSERT_TRUE(KuznetsovArtyomTbb::isEqual(resSeq[i], outputMatr[i]));
  }
}

TEST(Kuznetsov_a_cannon_matr_mult_tbb_perf_tests, test_800x800) {
  // Create data
  size_t size = 800;
  size_t block = 400;

  double minVal = -100.0;
  double maxVal = 100.0;

  auto inputMatrOne = KuznetsovArtyomTbb::getRandomSquareMatrix(size, minVal, maxVal);
  auto inputMatrTwo = KuznetsovArtyomTbb::getRandomSquareMatrix(size, minVal, maxVal);
  std::vector<double> outputMatr(size * size, 0.0);

  // Create TaskData
  auto taskDataTbb = std::make_shared<ppc::core::TaskData>();

  // Add matrOne
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrOne.data()));
  taskDataTbb->inputs_count.emplace_back(inputMatrOne.size());

  // Add matrTwo
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrTwo.data()));
  taskDataTbb->inputs_count.emplace_back(inputMatrTwo.size());

  // Add size
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));

  // Add block
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block));

  // Add out matr
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputMatr.data()));
  taskDataTbb->outputs_count.emplace_back(outputMatr.size());

  auto resSeq = KuznetsovArtyomTbb::CannonMatrixMultSeq(inputMatrOne, inputMatrTwo, size, block);

  // Create Task
  auto testTaskTbb = std::make_shared<KuznetsovArtyomTbb::KuznetsovCannonMatrMultTbb>(taskDataTbb);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskTbb);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  auto resSize = resSeq.size();

  for (size_t i = 0; i < resSize; ++i) {
    ASSERT_TRUE(KuznetsovArtyomTbb::isEqual(resSeq[i], outputMatr[i]));
  }
}
