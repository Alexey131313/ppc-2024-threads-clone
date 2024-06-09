// Copyright 2023 Lesnikov Nikita
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/lesnikov_nikita_binary_labelling/include/ops_seq.hpp"

std::pair<std::vector<uint8_t>, int> createBigOddSizeImage(int m, int n, std::vector<uint8_t>* labelled) {
  if (m % 2 == 0 && n % 2 == 0) {
    return std::make_pair(std::vector<uint8_t>(), -1);
  }
  std::vector<uint8_t> result(m * n, 1);
  *labelled = std::vector<uint8_t>(m * n);
  int objectsNum = 0;
  for (int i = 0; i < m * n; i++) {
    if (i % 2 == 0) {
      result[i] = 0;
      objectsNum++;
      (*labelled)[i] = objectsNum;
    }
  }
  return std::make_pair(result, objectsNum);
}

TEST(sequential_binary_labelling_test, test_pipeline_run) {
  int m = 131;
  int n = 131;
  auto serializedM = BinaryLabellingSequential::serializeInt32(m);
  auto serializedN = BinaryLabellingSequential::serializeInt32(n);
  std::vector<uint8_t> expected;
  auto res = createBigOddSizeImage(m, n, &expected);
  std::vector<uint8_t> in = res.first;
  std::vector<uint8_t> outV(in.size());
  std::vector<uint8_t> outNum(4);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->state_of_testing = ppc::core::TaskData::FUNC;
  taskDataSeq->inputs.push_back(in.data());
  taskDataSeq->inputs.push_back(serializedM.data());
  taskDataSeq->inputs.push_back(serializedN.data());
  taskDataSeq->inputs_count.push_back(in.size());
  taskDataSeq->inputs_count.push_back(4);
  taskDataSeq->inputs_count.push_back(4);
  taskDataSeq->outputs.push_back(outV.data());
  taskDataSeq->outputs.push_back(outNum.data());
  taskDataSeq->outputs_count.push_back(outV.size());
  taskDataSeq->outputs_count.push_back(4);

  auto binaryLabellingSequential = std::make_shared<BinaryLabellingSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(binaryLabellingSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  uint32_t expectedObjectsNum = res.second;

  EXPECT_EQ(outV, expected);
  EXPECT_EQ(BinaryLabellingSequential::deserializeInt32(outNum.data()), expectedObjectsNum);
}

TEST(sequential_binary_labelling_test, test_task_run) {
  int m = 131;
  int n = 131;
  auto serializedM = BinaryLabellingSequential::serializeInt32(m);
  auto serializedN = BinaryLabellingSequential::serializeInt32(n);
  std::vector<uint8_t> expected;
  auto res = createBigOddSizeImage(m, n, &expected);
  std::vector<uint8_t> in = res.first;
  std::vector<uint8_t> outV(in.size());
  std::vector<uint8_t> outNum(4);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->state_of_testing = ppc::core::TaskData::FUNC;
  taskDataSeq->inputs.push_back(in.data());
  taskDataSeq->inputs.push_back(serializedM.data());
  taskDataSeq->inputs.push_back(serializedN.data());
  taskDataSeq->inputs_count.push_back(in.size());
  taskDataSeq->inputs_count.push_back(4);
  taskDataSeq->inputs_count.push_back(4);
  taskDataSeq->outputs.push_back(outV.data());
  taskDataSeq->outputs.push_back(outNum.data());
  taskDataSeq->outputs_count.push_back(outV.size());
  taskDataSeq->outputs_count.push_back(4);

  auto binaryLabellingSequential = std::make_shared<BinaryLabellingSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(binaryLabellingSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  uint32_t expectedObjectsNum = res.second;

  EXPECT_EQ(outV, expected);
  EXPECT_EQ(BinaryLabellingSequential::deserializeInt32(outNum.data()), expectedObjectsNum);
}
