// Copyright 2024 Kuznetsov Artem
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/kuznetsov_a_cannon_matr_mult/include/ops_omp.hpp"

TEST(Kuznetsov_a_cannon_matr_mult_omp_perf_tests, test_900x900) {
  // Create data
  size_t size = 900;
  size_t block = 450;

  double minVal = -100.0;
  double maxVal = 100.0;

  auto inputMatrOne = KuznetsovArtyomOmp::getRandomSquareMatrix(size, minVal, maxVal);
  auto inputMatrTwo = KuznetsovArtyomOmp::getRandomSquareMatrix(size, minVal, maxVal);
  std::vector<double> outputMatr(size * size, 0.0);

  // Create TaskData
  auto taskDataOmp = std::make_shared<ppc::core::TaskData>();

  // Add matrOne
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrOne.data()));
  taskDataOmp->inputs_count.emplace_back(inputMatrOne.size());

  // Add matrTwo
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrTwo.data()));
  taskDataOmp->inputs_count.emplace_back(inputMatrTwo.size());

  // Add size
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));

  // Add block
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block));

  // Add out matr
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputMatr.data()));
  taskDataOmp->outputs_count.emplace_back(outputMatr.size());

  auto resSeq = KuznetsovArtyomOmp::CannonMatrixMultSeq(inputMatrOne, inputMatrTwo, size, block);

  // Create Task
  auto testTaskOmp = std::make_shared<KuznetsovArtyomOmp::KuznetsovCannonMatrMultOmp>(taskDataOmp);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOmp);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  auto resSize = resSeq.size();

  for (size_t i = 0; i < resSize; ++i) {
    ASSERT_TRUE(KuznetsovArtyomOmp::isEqual(resSeq[i], outputMatr[i]));
  }
}

TEST(Kuznetsov_a_cannon_matr_mult_omp_perf_tests, test_800x800) {
  // Create data
  size_t size = 800;
  size_t block = 400;

  double minVal = -100.0;
  double maxVal = 100.0;

  auto inputMatrOne = KuznetsovArtyomOmp::getRandomSquareMatrix(size, minVal, maxVal);
  auto inputMatrTwo = KuznetsovArtyomOmp::getRandomSquareMatrix(size, minVal, maxVal);
  std::vector<double> outputMatr(size * size, 0.0);

  // Create TaskData
  auto taskDataOmp = std::make_shared<ppc::core::TaskData>();

  // Add matrOne
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrOne.data()));
  taskDataOmp->inputs_count.emplace_back(inputMatrOne.size());

  // Add matrTwo
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrTwo.data()));
  taskDataOmp->inputs_count.emplace_back(inputMatrTwo.size());

  // Add size
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));

  // Add block
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block));

  // Add out matr
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputMatr.data()));
  taskDataOmp->outputs_count.emplace_back(outputMatr.size());

  auto resSeq = KuznetsovArtyomOmp::CannonMatrixMultSeq(inputMatrOne, inputMatrTwo, size, block);

  // Create Task
  auto testTaskOmp = std::make_shared<KuznetsovArtyomOmp::KuznetsovCannonMatrMultOmp>(taskDataOmp);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskOmp);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  auto resSize = resSeq.size();

  for (size_t i = 0; i < resSize; ++i) {
    ASSERT_TRUE(KuznetsovArtyomOmp::isEqual(resSeq[i], outputMatr[i]));
  }
}
