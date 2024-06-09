// Copyright 2024 Eremin Alexander
#include <gtest/gtest.h>

#include <vector>

#include "tbb/eremin_a_int_radixsort/include/ops_seq.hpp"

TEST(eremin_a_int_radixsort_tbb, simple_test) {
  // Create data
  std::vector<int> in = {5, 4, 3};
  std::vector<int> answer = {3, 4, 5};
  std::vector<int> out(3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortTaskTBB Task(taskDataSeq);
  ASSERT_EQ(Task.validation(), true);
  Task.pre_processing();
  Task.run();
  Task.post_processing();
  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], answer[i]);
  }
}

TEST(eremin_a_int_radixsort_tbb, simple_test_with_six_values) {
  // Create data
  std::vector<int> in = {5, 1, 3, 6, 9, 0};
  std::vector<int> answer = {0, 1, 3, 5, 6, 9};
  std::vector<int> out(6);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortTaskTBB Task(taskDataSeq);
  ASSERT_EQ(Task.validation(), true);
  Task.pre_processing();
  Task.run();
  Task.post_processing();
  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], answer[i]);
  }
}

TEST(eremin_a_int_radixsort_tbb, simple_test_with_eleven_values) {
  // Create data
  std::vector<int> in = {5, 4, 3, 2, 1, 20, 17, 8, 10, 16, 15};
  std::vector<int> answer = {1, 2, 3, 4, 5, 8, 10, 15, 16, 17, 20};
  std::vector<int> out(11);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortTaskTBB Task(taskDataSeq);
  ASSERT_EQ(Task.validation(), true);
  Task.pre_processing();
  Task.run();
  Task.post_processing();
  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], answer[i]);
  }
}

TEST(eremin_a_int_radixsort_tbb, simple_test_with_duplicate) {
  // Create data
  std::vector<int> in = {1, 10, 10, 8, 7, 6, 5, 4, 3, 2, 1, 7, 6};
  std::vector<int> answer = {1, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 10, 10};
  std::vector<int> out(13);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortTaskTBB Task(taskDataSeq);
  ASSERT_EQ(Task.validation(), true);
  Task.pre_processing();
  Task.run();
  Task.post_processing();
  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], answer[i]);
  }
}

TEST(eremin_a_int_radixsort_tbb, sort_random_vector) {
  // Create data
  std::vector<int> in = {64, 3, 99, 63, 57, 71, 53, 76, 85, 42, 58, 51, 63, 6, 75, 56, 84, 8, 6, 12};
  std::vector<int> answer = {3, 6, 6, 8, 12, 42, 51, 53, 56, 57, 58, 63, 63, 64, 71, 75, 76, 84, 85, 99};
  std::vector<int> out(20);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortTaskTBB Task(taskDataSeq);
  ASSERT_EQ(Task.validation(), true);
  Task.pre_processing();
  Task.run();
  Task.post_processing();
  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], answer[i]);
  }
}

TEST(eremin_a_int_radixsort_tbb, validation_test_one) {
  // Create data
  std::vector<int> in = {3, 2, 1};
  std::vector<int> answer = {1, 2, 3};
  std::vector<int> out(4);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  RadixSortTaskTBB Task(taskDataSeq);
  ASSERT_NE(Task.validation(), true);
}

TEST(eremin_a_int_radixsort_tbb, validation_test_two) {
  // Create data
  std::vector<int> in;
  std::vector<int> answer = {1, 2, 3};
  std::vector<int> out(0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  RadixSortTaskTBB Task(taskDataSeq);
  ASSERT_NE(Task.validation(), true);
}