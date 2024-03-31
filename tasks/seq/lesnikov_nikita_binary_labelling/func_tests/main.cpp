// Copyright 2023 Lesnikov Nikita
#include <gtest/gtest.h>

#include <vector>
#include <iostream>

#include "seq/lesnikov_nikita_binary_labelling/include/ops_seq.hpp"

TEST(Sequential, ZeroMatrixTest) {
  int m = 2;
  int n = 2;
  auto serializedM = BinaryLabellingSequential::serializeInt32(m);
  auto serializedN = BinaryLabellingSequential::serializeInt32(n);
  std::vector<uint8_t> in = {0, 0, 0, 0};
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

  BinaryLabellingSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<uint8_t> expected = {0, 0, 0, 0};
  uint32_t expectedObjectsNum = 0;

  std::cout << "obj nums: " << BinaryLabellingSequential::deserializeInt32(outNum.data()) << " " << expectedObjectsNum << std::endl;

  BinaryLabellingSequential::visualize(outV, m, n);

  EXPECT_EQ(outV, expected);
  EXPECT_EQ(BinaryLabellingSequential::deserializeInt32(outNum.data()), expectedObjectsNum);
}

TEST(SequentialBinaryLabelling, all1Test) {
  int m = 2;
  int n = 2;
  auto serializedM = BinaryLabellingSequential::serializeInt32(m);
  auto serializedN = BinaryLabellingSequential::serializeInt32(n);
  std::vector<uint8_t> in = {1, 1, 1, 1};
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

  BinaryLabellingSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<uint8_t> expected = {1, 1, 1, 1};
  uint32_t expectedObjectsNum = 1;

  std::cout << "obj nums: " << BinaryLabellingSequential::deserializeInt32(outNum.data()) << " " << expectedObjectsNum
            << std::endl;

  BinaryLabellingSequential::visualize(outV, m, n);

  EXPECT_EQ(outV, expected);
  EXPECT_EQ(BinaryLabellingSequential::deserializeInt32(outNum.data()), expectedObjectsNum);
}

TEST(SequentialBinaryLabelling, allSeperatedTest) { 
     int m = 5;
  int n = 5;
  auto serializedM = BinaryLabellingSequential::serializeInt32(m);
  auto serializedN = BinaryLabellingSequential::serializeInt32(n);
  std::vector<uint8_t> in = {
      1, 0, 1, 0, 1,
      0, 1, 0, 1, 0,
      1, 0, 1, 0, 1,
      0, 1, 0, 1, 0,
      1, 0, 1, 0, 1};
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

  BinaryLabellingSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<uint8_t> expected = {
      1, 0, 2, 0, 3, 
      0, 4, 0, 5, 0,
      6, 0, 7, 0, 8, 
      0, 9, 0, 10, 0, 
      11, 0, 12, 0, 13};
  uint32_t expectedObjectsNum = 13;

  std::cout << "obj nums: " << BinaryLabellingSequential::deserializeInt32(outNum.data()) << " " << expectedObjectsNum
            << std::endl;

  BinaryLabellingSequential::visualize(outV, m, n);

  EXPECT_EQ(outV, expected);
  EXPECT_EQ(BinaryLabellingSequential::deserializeInt32(outNum.data()), expectedObjectsNum);
}

TEST(SequentialBinaryLabelling, Object2Test) {
  int m = 5;
  int n = 5;
  auto serializedM = BinaryLabellingSequential::serializeInt32(m);
  auto serializedN = BinaryLabellingSequential::serializeInt32(n);
  std::vector<uint8_t> in = {
      1, 1, 1, 0, 0,
      1, 1, 0, 1, 0,
      1, 0, 1, 1, 1,
      0, 0, 1, 0, 1,
      0, 0, 1, 1, 1};
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

  BinaryLabellingSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<uint8_t> expected = {
      1, 1, 1, 0, 0, 
      1, 1, 0, 2, 0, 
      1, 0, 2, 2, 2, 
      0, 0, 2, 0, 2, 
      0, 0, 2, 2, 2};
  uint32_t expectedObjectsNum = 2;

  std::cout << "obj nums: " << BinaryLabellingSequential::deserializeInt32(outNum.data()) << " " << expectedObjectsNum
            << std::endl;

  BinaryLabellingSequential::visualize(outV, m, n);

  EXPECT_EQ(outV, expected);
  EXPECT_EQ(BinaryLabellingSequential::deserializeInt32(outNum.data()), expectedObjectsNum);
}

TEST(SequentialBinaryLabelling, dummy4) { EXPECT_TRUE(true); }
