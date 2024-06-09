// Copyright 2024 Smirnov Leonid
#include <gtest/gtest.h>

#include <vector>

#include "seq/smirnov_l_radix_sort/include/ops_seq.hpp"

TEST(Smirnov_L_Radix_Sort_Test, Single_Element_Vector_Test) {
  // Create data
  std::vector<int> in = {1};
  std::vector<int> expected = {1};
  std::vector<int> out(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortSequential radixSortSequential(taskDataSeq);
  ASSERT_EQ(radixSortSequential.validation(), true);
  ASSERT_EQ(radixSortSequential.pre_processing(), true);
  ASSERT_EQ(radixSortSequential.run(), true);
  ASSERT_EQ(radixSortSequential.post_processing(), true);
  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], expected[i]);
  }
}

TEST(Smirnov_L_Radix_Sort_Test, Two_Element_Vector_Test) {
  // Create data
  std::vector<int> in = {2, 1};
  std::vector<int> expected = {1, 2};
  std::vector<int> out(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortSequential radixSortSequential(taskDataSeq);
  ASSERT_EQ(radixSortSequential.validation(), true);
  ASSERT_EQ(radixSortSequential.pre_processing(), true);
  ASSERT_EQ(radixSortSequential.run(), true);
  ASSERT_EQ(radixSortSequential.post_processing(), true);

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], expected[i]);
  }
}

TEST(Smirnov_L_Radix_Sort_Test, Three_Element_Vector_Test) {
  // Create data
  std::vector<int> in = {6, 3, 1};
  std::vector<int> expected = {1, 3, 6};
  std::vector<int> out(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortSequential radixSortSequential(taskDataSeq);
  ASSERT_EQ(radixSortSequential.validation(), true);
  ASSERT_EQ(radixSortSequential.pre_processing(), true);
  ASSERT_EQ(radixSortSequential.run(), true);
  ASSERT_EQ(radixSortSequential.post_processing(), true);

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], expected[i]);
  }
}

TEST(Smirnov_L_Radix_Sort_Test, Ten_Element_Vector_Test) {
  // Create data
  std::vector<int> in = {6, 3, 1, 30, 50, 113, 5, 17, 76, 21};
  std::vector<int> expected = {1, 3, 5, 6, 17, 21, 30, 50, 76, 113};
  std::vector<int> out(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortSequential radixSortSequential(taskDataSeq);
  ASSERT_EQ(radixSortSequential.validation(), true);
  ASSERT_EQ(radixSortSequential.pre_processing(), true);
  ASSERT_EQ(radixSortSequential.run(), true);
  ASSERT_EQ(radixSortSequential.post_processing(), true);

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], expected[i]);
  }
}

TEST(Smirnov_L_Radix_Sort_Test, Validation_One_Test) {
  // Create data
  std::vector<int> in;
  std::vector<int> out(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortSequential radixSortSequential(taskDataSeq);
  ASSERT_FALSE(radixSortSequential.validation());
}

TEST(Smirnov_L_Radix_Sort_Test, Validation_Two_Test) {
  // Create data
  std::vector<int> in1 = {1, 2, 3};
  std::vector<int> in2 = {3, 2, 1};
  std::vector<int> out(in1.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortSequential radixSortSequential(taskDataSeq);
  ASSERT_FALSE(radixSortSequential.validation());
}

TEST(Smirnov_L_Radix_Sort_Test, Validation_Three_Test) {
  // Create data
  std::vector<int> in = {9, 6, 3};
  std::vector<int> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortSequential radixSortSequential(taskDataSeq);
  ASSERT_FALSE(radixSortSequential.validation());
}

TEST(Smirnov_L_Radix_Sort_Test, Validation_Four_Test) {
  // Create data
  std::vector<int> in = {9, 6, 3};
  std::vector<int> out(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(10);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortSequential radixSortSequential(taskDataSeq);
  ASSERT_FALSE(radixSortSequential.validation());
}

TEST(Smirnov_L_Radix_Sort_Test, Validation_Five_Test) {
  // Create data
  std::vector<int> in = {9, 6, 3};
  std::vector<int> out(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->outputs_count.emplace_back(10);

  // Create Task
  RadixSortSequential radixSortSequential(taskDataSeq);
  ASSERT_FALSE(radixSortSequential.validation());
}

TEST(Smirnov_L_Radix_Sort_Test, Validation_Six_Test) {
  // Create data
  std::vector<int> in = {1, 2, 3};
  std::vector<int> out1(in.size(), 0);
  std::vector<int> out2(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
  taskDataSeq->outputs_count.emplace_back(out1.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataSeq->outputs_count.emplace_back(out2.size());

  // Create Task
  RadixSortSequential radixSortSequential(taskDataSeq);
  ASSERT_FALSE(radixSortSequential.validation());
}

TEST(Smirnov_L_Radix_Sort_Test, Validation_Seven_Test) {
  // Create data
  std::vector<int> in = {1, 2, 3};
  std::vector<int> out(4, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortSequential radixSortSequential(taskDataSeq);
  ASSERT_FALSE(radixSortSequential.validation());
}
