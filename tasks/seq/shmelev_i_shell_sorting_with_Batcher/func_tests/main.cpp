// Copyright 2024 Shmelev Ivan
#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "seq/shmelev_i_shell_sorting_with_Batcher/include/ops_seq.hpp"

TEST(shmelev_i_shell_sorting_with_Batcher, validation_check) {
  std::vector<int> array = {10, -1, 0};
  std::vector<int> sorted_array(3);
  std::vector<int> expected = {-1, 0, 10};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&array));
  taskDataSeq->inputs_count.emplace_back(array.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&sorted_array));
  taskDataSeq->outputs_count.emplace_back(expected.size() + 1);

  // Create Task
  ShmelevTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(shmelev_i_shell_sorting_with_Batcher, Sorting_small_array) {
  std::vector<int> array = {12, 0, -32};
  std::vector<int> sorted_array(3);
  std::vector<int> expected = {-32, 0, 12};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&array));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&sorted_array));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  ShmelevTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(sorted_array[i], expected[i]);
  }
}

TEST(shmelev_i_shell_sorting_with_Batcher, Sorting_another_small_array) {
  std::vector<int> array = {60, -11, 234, 10};
  std::vector<int> sorted_array(4);
  std::vector<int> expected = {-11, 10, 60, 234};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&array));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&sorted_array));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  ShmelevTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(sorted_array[i], expected[i]);
  }
}

TEST(shmelev_i_shell_sorting_with_Batcher, Sorting_medium_array) {
  std::vector<int> array = {17, 23, 8, 14, 29, 5, 11, 20, 26, 2, 32, 35, 10, 38, 41, 47};
  std::vector<int> sorted_array(16);
  std::vector<int> expected = {2, 5, 8, 10, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 47};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&array));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&sorted_array));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  ShmelevTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(sorted_array[i], expected[i]);
  }
}

TEST(shmelev_i_shell_sorting_with_Batcher, Sorting_big_array) {
  std::vector<int> array = {33, 56, 45, 30, 11, 47, 62, 19, 36, 58, 51, 14, 40, 8,  27,
                            34, 48, 24, 25, 39, 13, 12, 55, 23, 59, 37, 17, 50, 42, 9};
  std::vector<int> sorted_array(32);
  std::vector<int> expected = {8,  9,  11, 12, 13, 14, 17, 19, 23, 24, 25, 27, 30, 33, 34,
                               36, 37, 39, 40, 42, 45, 47, 48, 50, 51, 55, 56, 58, 59, 62};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&array));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&sorted_array));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  ShmelevTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (int i = 0; i < 30; i++) {
    EXPECT_EQ(sorted_array[i], expected[i]);
  }
}
