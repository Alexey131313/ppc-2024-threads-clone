// Copyright 2024 Chuvashov Andrey

#include <gtest/gtest.h>

#include <vector>

#include "seq/chuvashov_a_batcher_ints_sort/include/batcher_ints_sort.hpp"

TEST(ChuvashovABatcherFuncTests, Test_Simple_Array) {
  std::vector<int> vect = {24, 56, 1, 91, 23, 19, 144, 8, -140, 19};
  std::vector<int> result(vect.size(), 0);

  std::shared_ptr<ppc::core::TaskData> seqData = std::make_shared<ppc::core::TaskData>();
  seqData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  seqData->inputs_count.emplace_back(vect.size());
  seqData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  seqData->outputs_count.emplace_back(result.size());

  Chuvashov_SequentialBatcherSort testSeq(seqData);
  ASSERT_EQ(testSeq.validation(), true);
  testSeq.pre_processing();
  testSeq.run();
  testSeq.post_processing();
  ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));
}

TEST(ChuvashovABatcherFuncTests, Test_100_Size) {
  std::vector<int> vect = Chuvashov_GenerateVector(100);
  std::vector<int> result(vect.size(), 0);

  std::shared_ptr<ppc::core::TaskData> seqData = std::make_shared<ppc::core::TaskData>();
  seqData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  seqData->inputs_count.emplace_back(vect.size());
  seqData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  seqData->outputs_count.emplace_back(result.size());

  Chuvashov_SequentialBatcherSort testSeq(seqData);
  ASSERT_EQ(testSeq.validation(), true);
  testSeq.pre_processing();
  testSeq.run();
  testSeq.post_processing();
  ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));
}

TEST(ChuvashovABatcherFuncTests, Test_200_Size) {
  std::vector<int> vect = Chuvashov_GenerateVector(200);
  std::vector<int> result(vect.size(), 0);

  std::shared_ptr<ppc::core::TaskData> seqData = std::make_shared<ppc::core::TaskData>();
  seqData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  seqData->inputs_count.emplace_back(vect.size());
  seqData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  seqData->outputs_count.emplace_back(result.size());

  Chuvashov_SequentialBatcherSort testSeq(seqData);
  ASSERT_EQ(testSeq.validation(), true);
  testSeq.pre_processing();
  testSeq.run();
  testSeq.post_processing();
  ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));
}

TEST(ChuvashovABatcherFuncTests, Test_Little_Size) {
  std::vector<int> vect = Chuvashov_GenerateVector(2);
  std::vector<int> result(vect.size(), 0);

  std::shared_ptr<ppc::core::TaskData> seqData = std::make_shared<ppc::core::TaskData>();
  seqData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  seqData->inputs_count.emplace_back(vect.size());
  seqData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  seqData->outputs_count.emplace_back(result.size());

  Chuvashov_SequentialBatcherSort testSeq(seqData);
  ASSERT_EQ(testSeq.validation(), true);
  testSeq.pre_processing();
  testSeq.run();
  testSeq.post_processing();
  ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));
}

TEST(ChuvashovABatcherFuncTests, Test_Only_Negative) {
  std::vector<int> vect = {-134, -200, -1, -23, -1000, -12};
  std::vector<int> result(vect.size(), 0);

  std::shared_ptr<ppc::core::TaskData> seqData = std::make_shared<ppc::core::TaskData>();
  seqData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  seqData->inputs_count.emplace_back(vect.size());
  seqData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  seqData->outputs_count.emplace_back(result.size());

  Chuvashov_SequentialBatcherSort testSeq(seqData);
  ASSERT_EQ(testSeq.validation(), true);
  testSeq.pre_processing();
  testSeq.run();
  testSeq.post_processing();
  ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));
}

TEST(ChuvashovABatcherFuncTests, Test_Only_Zeroes) {
  std::vector<int> vect = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> result(vect.size(), 0);

  std::shared_ptr<ppc::core::TaskData> seqData = std::make_shared<ppc::core::TaskData>();
  seqData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  seqData->inputs_count.emplace_back(vect.size());
  seqData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  seqData->outputs_count.emplace_back(result.size());

  Chuvashov_SequentialBatcherSort testSeq(seqData);
  ASSERT_EQ(testSeq.validation(), true);
  testSeq.pre_processing();
  testSeq.run();
  testSeq.post_processing();
  ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));
}

TEST(ChuvashovABatcherFuncTests, Test_Already_Sorted) {
  std::vector<int> vect = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> result(vect.size(), 0);

  std::shared_ptr<ppc::core::TaskData> seqData = std::make_shared<ppc::core::TaskData>();
  seqData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
  seqData->inputs_count.emplace_back(vect.size());
  seqData->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  seqData->outputs_count.emplace_back(result.size());

  Chuvashov_SequentialBatcherSort testSeq(seqData);
  ASSERT_EQ(testSeq.validation(), true);
  testSeq.pre_processing();
  testSeq.run();
  testSeq.post_processing();
  ASSERT_TRUE(std::is_sorted(result.begin(), result.end()));
}
