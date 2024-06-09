// Copyright 2024 Zakharov Artem
#include <gtest/gtest.h>

#include <vector>

#include "seq/zakharov_a_radix_sort/include/ops_seq.hpp"

TEST(Zakharov_a_radix_sort_seq, Positive) {
  const int count = 7;

  // Create data
  std::vector<int> in = {2, 1, 10, 4, 15, 100, 52};
  std::vector<int> out(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ZakharovRadixSortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::vector<int> result = {1, 2, 4, 10, 15, 52, 100};
  ASSERT_EQ(result, out);
}

TEST(Zakharov_a_radix_sort_seq, Negative) {
  const int count = 10;

  // Create data
  std::vector<int> in = {-2, -15, -1000, -324, -176, -127, -52, -19, -17, -3};
  std::vector<int> out(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ZakharovRadixSortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::vector<int> result = {-1000, -324, -176, -127, -52, -19, -17, -15, -3, -2};
  ASSERT_EQ(result, out);
}

TEST(Zakharov_a_radix_sort_seq, Sorted) {
  const int count = 11;

  // Create data
  std::vector<int> in = {0, 1, 2, 3, 4, 5, 6, 10, 15, 200, 20000};
  std::vector<int> out(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ZakharovRadixSortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::vector<int> result = {0, 1, 2, 3, 4, 5, 6, 10, 15, 200, 20000};
  ASSERT_EQ(result, out);
}

TEST(Zakharov_a_radix_sort_seq, ReversSorted) {
  const int count = 9;

  // Create data
  std::vector<int> in = {1000, 68, 44, 32, 22, 15, 13, 5, 1};
  std::vector<int> out(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ZakharovRadixSortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::vector<int> result = {1, 5, 13, 15, 22, 32, 44, 68, 1000};
  ASSERT_EQ(result, out);
}

TEST(Zakharov_a_radix_sort_seq, DuplicateValues) {
  const int count = 12;

  // Create data
  std::vector<int> in = {-25, 1, -25, 43, 0, -23, -55, 66, 66, 3, 2, 0};
  std::vector<int> out(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ZakharovRadixSortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::vector<int> result = {-55, -25, -25, -23, 0, 0, 1, 2, 3, 43, 66, 66};
  ASSERT_EQ(result, out);
}

TEST(Zakharov_a_radix_sort_seq, IdenticalElements) {
  const int count = 6;

  // Create data
  std::vector<int> in = {5, 5, 5, 5, 5, 5};
  std::vector<int> out(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ZakharovRadixSortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::vector<int> result = {5, 5, 5, 5, 5, 5};
  ASSERT_EQ(result, out);
}
