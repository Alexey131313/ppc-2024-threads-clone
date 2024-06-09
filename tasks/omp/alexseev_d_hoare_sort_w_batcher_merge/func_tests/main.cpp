// Copyright 2024 Alexseev Danila
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "omp/alexseev_d_hoare_sort_w_batcher_merge/include/hoare_sort_w_batcher_merge_omp.hpp"

using namespace alexseev_omp;

TEST(alexseev_d_hoare_sort_w_batcher_merge_omp, Random_Constatnt_Array) {
  std::vector<int> inputArray = {3, 5, 4, 6, 9, 2, 8, 1, 7, 0};

  // Sequential
  std::vector<int> outputArraySeq(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArraySeq.data()));
  hoareSortWBatcherMergeSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeSequential hoareSortWBatcherMergeSequential(hoareSortWBatcherMergeSeq);
  ASSERT_TRUE(hoareSortWBatcherMergeSequential.validation());
  hoareSortWBatcherMergeSequential.pre_processing();
  hoareSortWBatcherMergeSequential.run();
  hoareSortWBatcherMergeSequential.post_processing();

  // OMP
  std::vector<int> outputArrayOMP(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeOMP = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeOMP->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeOMP->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeOMP->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArrayOMP.data()));
  hoareSortWBatcherMergeOMP->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeOMP hoareSortWBatcherMergeOMPParallel(hoareSortWBatcherMergeOMP);
  ASSERT_TRUE(hoareSortWBatcherMergeOMPParallel.validation());
  hoareSortWBatcherMergeOMPParallel.pre_processing();
  hoareSortWBatcherMergeOMPParallel.run();
  hoareSortWBatcherMergeOMPParallel.post_processing();

  ASSERT_EQ(outputArrayOMP, outputArraySeq);
}

TEST(alexseev_d_hoare_sort_w_batcher_merge_omp, Random_Array_Even) {
  std::vector<int> inputArray = {373,  -119, -730, -699, 680, 538,  462, -844, 530,  -266, 734,  257,  -482,
                                 670,  387,  -618, 705,  808, 120,  634, -294, -617, 733,  109,  10,   -631,
                                 -780, 719,  -156, -155, 945, -743, 552, -654, -774, -133, -331, -508, -827,
                                 64,   -239, 792,  693,  383, -963, 954, -503, 932,  -499, 885};
  ;

  // Sequential
  std::vector<int> outputArraySeq(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArraySeq.data()));
  hoareSortWBatcherMergeSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeSequential hoareSortWBatcherMergeSequential(hoareSortWBatcherMergeSeq);
  ASSERT_TRUE(hoareSortWBatcherMergeSequential.validation());
  hoareSortWBatcherMergeSequential.pre_processing();
  hoareSortWBatcherMergeSequential.run();
  hoareSortWBatcherMergeSequential.post_processing();

  // OMP
  std::vector<int> outputArrayOMP(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeOMP = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeOMP->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeOMP->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeOMP->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArrayOMP.data()));
  hoareSortWBatcherMergeOMP->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeOMP hoareSortWBatcherMergeOMPParallel(hoareSortWBatcherMergeOMP);
  ASSERT_TRUE(hoareSortWBatcherMergeOMPParallel.validation());
  hoareSortWBatcherMergeOMPParallel.pre_processing();
  hoareSortWBatcherMergeOMPParallel.run();
  hoareSortWBatcherMergeOMPParallel.post_processing();

  ASSERT_EQ(outputArrayOMP, outputArraySeq);
}

TEST(alexseev_d_hoare_sort_w_batcher_merge_omp, Random_Array_Odd) {
  std::vector<int> inputArray = {405,  -937, 839,  122,  -42,  777,  406,  -205, 520, 824,  -384, 463,  738,
                                 -887, -444, 676,  -180, -933, -44,  663,  -58,  829, 873,  267,  740,  -297,
                                 516,  -61,  -306, 494,  615,  571,  -460, 742,  142, 738,  -46,  -263, 10,
                                 946,  -332, 232,  101,  -773, -626, 124,  -936, -42, -740, -698, -896};
  ;

  // Sequential
  std::vector<int> outputArraySeq(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArraySeq.data()));
  hoareSortWBatcherMergeSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeSequential hoareSortWBatcherMergeSequential(hoareSortWBatcherMergeSeq);
  ASSERT_TRUE(hoareSortWBatcherMergeSequential.validation());
  hoareSortWBatcherMergeSequential.pre_processing();
  hoareSortWBatcherMergeSequential.run();
  hoareSortWBatcherMergeSequential.post_processing();

  // OMP
  std::vector<int> outputArrayOMP(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeOMP = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeOMP->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeOMP->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeOMP->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArrayOMP.data()));
  hoareSortWBatcherMergeOMP->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeOMP hoareSortWBatcherMergeOMPParallel(hoareSortWBatcherMergeOMP);
  ASSERT_TRUE(hoareSortWBatcherMergeOMPParallel.validation());
  hoareSortWBatcherMergeOMPParallel.pre_processing();
  hoareSortWBatcherMergeOMPParallel.run();
  hoareSortWBatcherMergeOMPParallel.post_processing();

  ASSERT_EQ(outputArrayOMP, outputArraySeq);
}

TEST(alexseev_d_hoare_sort_w_batcher_merge_omp, Unit_Array) {
  std::vector<int> inputArray = {-307};

  // Sequential
  std::vector<int> outputArraySeq(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArraySeq.data()));
  hoareSortWBatcherMergeSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeSequential hoareSortWBatcherMergeSequential(hoareSortWBatcherMergeSeq);
  ASSERT_TRUE(hoareSortWBatcherMergeSequential.validation());
  hoareSortWBatcherMergeSequential.pre_processing();
  hoareSortWBatcherMergeSequential.run();
  hoareSortWBatcherMergeSequential.post_processing();

  // OMP
  std::vector<int> outputArrayOMP(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeOMP = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeOMP->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeOMP->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeOMP->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArrayOMP.data()));
  hoareSortWBatcherMergeOMP->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeOMP hoareSortWBatcherMergeOMPParallel(hoareSortWBatcherMergeOMP);
  ASSERT_TRUE(hoareSortWBatcherMergeOMPParallel.validation());
  hoareSortWBatcherMergeOMPParallel.pre_processing();
  hoareSortWBatcherMergeOMPParallel.run();
  hoareSortWBatcherMergeOMPParallel.post_processing();

  ASSERT_EQ(outputArrayOMP, outputArraySeq);
}

TEST(alexseev_d_hoare_sort_w_batcher_merge_omp, Empty_Array) {
  std::vector<int> inputArray = {};

  // Sequential
  std::vector<int> outputArraySeq(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeSeq = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeSeq->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArraySeq.data()));
  hoareSortWBatcherMergeSeq->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeSequential hoareSortWBatcherMergeSequential(hoareSortWBatcherMergeSeq);
  ASSERT_TRUE(hoareSortWBatcherMergeSequential.validation());
  hoareSortWBatcherMergeSequential.pre_processing();
  hoareSortWBatcherMergeSequential.run();
  hoareSortWBatcherMergeSequential.post_processing();

  // OMP
  std::vector<int> outputArrayOMP(inputArray.size());

  std::shared_ptr<ppc::core::TaskData> hoareSortWBatcherMergeOMP = std::make_shared<ppc::core::TaskData>();
  hoareSortWBatcherMergeOMP->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  hoareSortWBatcherMergeOMP->inputs_count.emplace_back(inputArray.size());
  hoareSortWBatcherMergeOMP->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputArrayOMP.data()));
  hoareSortWBatcherMergeOMP->outputs_count.emplace_back(inputArray.size());

  HoareSortWBatcherMergeOMP hoareSortWBatcherMergeOMPParallel(hoareSortWBatcherMergeOMP);
  ASSERT_TRUE(hoareSortWBatcherMergeOMPParallel.validation());
  hoareSortWBatcherMergeOMPParallel.pre_processing();
  hoareSortWBatcherMergeOMPParallel.run();
  hoareSortWBatcherMergeOMPParallel.post_processing();

  ASSERT_EQ(outputArrayOMP, outputArraySeq);
}