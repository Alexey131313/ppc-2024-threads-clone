// Copyright 2024 Alexseev Danila
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "seq/alexseev_d_hoare_sort_w_batcher_merge/include/hoare_sort_w_batcher_merge_seq.hpp"

using namespace alexseev_seq;

TEST(alexseev_d_hoare_sort_w_batcher_merge_seq, Random_Constatnt_Array) {
  std::vector<int> inputArray = {3, 5, 4, 6, 9, 2, 8, 1, 7, 0};
  std::vector<int> outputArray(inputArray.size());
  std::vector<int> expectedArray = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArray.data()));
  hoareSortWBatcherMergeSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeSequential hoareSortWBatcherMergeSequential(hoareSortWBatcherMergeSeq);
  ASSERT_TRUE(hoareSortWBatcherMergeSequential.validation());
  hoareSortWBatcherMergeSequential.pre_processing();
  hoareSortWBatcherMergeSequential.run();
  hoareSortWBatcherMergeSequential.post_processing();
  ASSERT_EQ(expectedArray, outputArray);
}

TEST(alexseev_d_hoare_sort_w_batcher_merge_seq, Random_Array_Even) {
  std::vector<int> inputArray = {373,  -119, -730, -699, 680, 538,  462, -844, 530,  -266, 734,  257,  -482,
                                 670,  387,  -618, 705,  808, 120,  634, -294, -617, 733,  109,  10,   -631,
                                 -780, 719,  -156, -155, 945, -743, 552, -654, -774, -133, -331, -508, -827,
                                 64,   -239, 792,  693,  383, -963, 954, -503, 932,  -499, 885};
  std::vector<int> sortedInputArray = inputArray;
  std::sort(sortedInputArray.begin(), sortedInputArray.end());
  std::vector<int> outputArray(inputArray.size());
  std::vector<int> expectedArray = sortedInputArray;

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArray.data()));
  hoareSortWBatcherMergeSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeSequential hoareSortWBatcherMergeSequential(hoareSortWBatcherMergeSeq);
  ASSERT_TRUE(hoareSortWBatcherMergeSequential.validation());
  hoareSortWBatcherMergeSequential.pre_processing();
  hoareSortWBatcherMergeSequential.run();
  hoareSortWBatcherMergeSequential.post_processing();
  ASSERT_EQ(expectedArray, outputArray);
}

TEST(alexseev_d_hoare_sort_w_batcher_merge_seq, Random_Array_Odd) {
  std::vector<int> inputArray = {405,  -937, 839,  122,  -42,  777,  406,  -205, 520, 824,  -384, 463,  738,
                                 -887, -444, 676,  -180, -933, -44,  663,  -58,  829, 873,  267,  740,  -297,
                                 516,  -61,  -306, 494,  615,  571,  -460, 742,  142, 738,  -46,  -263, 10,
                                 946,  -332, 232,  101,  -773, -626, 124,  -936, -42, -740, -698, -896};
  std::vector<int> sortedInputArray = inputArray;
  std::sort(sortedInputArray.begin(), sortedInputArray.end());
  std::vector<int> outputArray(inputArray.size());
  std::vector<int> expectedArray = sortedInputArray;

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArray.data()));
  hoareSortWBatcherMergeSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeSequential hoareSortWBatcherMergeSequential(hoareSortWBatcherMergeSeq);
  ASSERT_TRUE(hoareSortWBatcherMergeSequential.validation());
  hoareSortWBatcherMergeSequential.pre_processing();
  hoareSortWBatcherMergeSequential.run();
  hoareSortWBatcherMergeSequential.post_processing();
  ASSERT_EQ(expectedArray, outputArray);
}

TEST(alexseev_d_hoare_sort_w_batcher_merge_seq, Unit_Array) {
  std::vector<int> inputArray = {-307};
  std::vector<int> sortedInputArray = inputArray;
  std::sort(sortedInputArray.begin(), sortedInputArray.end());
  std::vector<int> outputArray(inputArray.size());
  std::vector<int> expectedArray = sortedInputArray;

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArray.data()));
  hoareSortWBatcherMergeSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeSequential hoareSortWBatcherMergeSequential(hoareSortWBatcherMergeSeq);
  ASSERT_TRUE(hoareSortWBatcherMergeSequential.validation());
  hoareSortWBatcherMergeSequential.pre_processing();
  hoareSortWBatcherMergeSequential.run();
  hoareSortWBatcherMergeSequential.post_processing();
  ASSERT_EQ(expectedArray, outputArray);
}

TEST(alexseev_d_hoare_sort_w_batcher_merge_seq, Empty_Array) {
  std::vector<int> inputArray = {};
  std::vector<int> outputArray(inputArray.size());
  std::vector<int> expectedArray = {};

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArray.data()));
  hoareSortWBatcherMergeSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeSequential hoareSortWBatcherMergeSequential(hoareSortWBatcherMergeSeq);
  ASSERT_TRUE(hoareSortWBatcherMergeSequential.validation());
  hoareSortWBatcherMergeSequential.pre_processing();
  hoareSortWBatcherMergeSequential.run();
  hoareSortWBatcherMergeSequential.post_processing();
  ASSERT_EQ(expectedArray, outputArray);
}