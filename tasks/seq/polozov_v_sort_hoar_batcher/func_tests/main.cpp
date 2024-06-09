// Copyright 2024 Polozov Vladislav
#include <gtest/gtest.h>

#include "seq/polozov_v_sort_hoar_batcher/include/ops_seq.hpp"

TEST(polozov_v_sort_hoar_batcher_seq, array_8) {
  // Create data
  std::vector<int> array = {4, 8, 1, 2, 9, 6, 5, 3};
  std::vector<int> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> DataSeq = std::make_shared<ppc::core::TaskData>();
  DataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  DataSeq->inputs_count.emplace_back(array.size());
  DataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  DataSeq->outputs_count.emplace_back(out.size());

  std::vector<int> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  SortHoarWithMergeBatcher myTask(DataSeq);
  ASSERT_EQ(myTask.validation(), true);
  myTask.pre_processing();
  myTask.run();
  myTask.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(polozov_v_sort_hoar_batcher_seq, rand_array_16) {
  // Create data
  std::vector<int> array = generate_data(16, 0, 16);
  std::vector<int> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> DataSeq = std::make_shared<ppc::core::TaskData>();
  DataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  DataSeq->inputs_count.emplace_back(array.size());
  DataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  DataSeq->outputs_count.emplace_back(out.size());

  std::vector<int> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  SortHoarWithMergeBatcher myTask(DataSeq);
  ASSERT_EQ(myTask.validation(), true);
  myTask.pre_processing();
  myTask.run();
  myTask.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(polozov_v_sort_hoar_batcher_seq, array_negative_16) {
  // Create data
  std::vector<int> array = generate_data(32, -16, -1);
  std::vector<int> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> DataSeq = std::make_shared<ppc::core::TaskData>();
  DataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  DataSeq->inputs_count.emplace_back(array.size());
  DataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  DataSeq->outputs_count.emplace_back(out.size());

  std::vector<int> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  SortHoarWithMergeBatcher myTask(DataSeq);
  ASSERT_EQ(myTask.validation(), true);
  myTask.pre_processing();
  myTask.run();
  myTask.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(polozov_v_sort_hoar_batcher_seq, array_32) {
  // Create data
  std::vector<int> array = generate_data(32, -100, 100);
  std::vector<int> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> DataSeq = std::make_shared<ppc::core::TaskData>();
  DataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  DataSeq->inputs_count.emplace_back(array.size());
  DataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  DataSeq->outputs_count.emplace_back(out.size());

  std::vector<int> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  SortHoarWithMergeBatcher myTask(DataSeq);
  ASSERT_EQ(myTask.validation(), true);
  myTask.pre_processing();
  myTask.run();
  myTask.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(polozov_v_sort_hoar_batcher_seq, array_4) {
  // Create data
  std::vector<int> array = generate_data(4, -10, 10);
  std::vector<int> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> DataSeq = std::make_shared<ppc::core::TaskData>();
  DataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  DataSeq->inputs_count.emplace_back(array.size());
  DataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  DataSeq->outputs_count.emplace_back(out.size());

  std::vector<int> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  SortHoarWithMergeBatcher myTask(DataSeq);
  ASSERT_EQ(myTask.validation(), true);
  myTask.pre_processing();
  myTask.run();
  myTask.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}