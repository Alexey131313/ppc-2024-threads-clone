// Copyright 2024 Benduyzhko Tatiana
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "seq/benduyzhko_t_shell_batcher/include/ops_seq.hpp"

TEST(benduyzhko_t_shell_batcher_seq, Test8Nums) {
  std::vector<int> arr = {-1, 2, 1, 4, 8, 6, 5, 10};
  std::vector<int> out = arr;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  BenduyzhkoSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}

TEST(benduyzhko_t_shell_batcher_seq, Test16Nums) {
  std::vector<int> arr = {-1, 2, 1, 4, 3, 0, 7, 9, 8, 6, 5, 10, -5, 12, -9, 11};
  std::vector<int> out = arr;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  BenduyzhkoSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}

TEST(benduyzhko_t_shell_batcher_seq, Test32NumsRandom) {
  int n = 32;
  std::vector<int> arr(n);
  BenduyzhkoSequential::get_random_numbers(arr.data(), n);
  std::vector<int> out = arr;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  BenduyzhkoSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}

TEST(benduyzhko_t_shell_batcher_seq, Test128NumsRandom) {
  int n = 128;
  std::vector<int> arr(n);
  BenduyzhkoSequential::get_random_numbers(arr.data(), n, -n, n);
  std::vector<int> out = arr;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  BenduyzhkoSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}

TEST(benduyzhko_t_shell_batcher_seq, Test512NumsRandom) {
  int n = 512;
  std::vector<int> arr(n);
  BenduyzhkoSequential::get_random_numbers(arr.data(), n, -n, n);
  std::vector<int> out = arr;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->inputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  BenduyzhkoSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}
