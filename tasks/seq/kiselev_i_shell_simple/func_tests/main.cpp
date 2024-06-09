// Copyright 2024 Kiselev Igor
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "seq/kiselev_i_shell_simple/include/shell_simple.hpp"

using namespace Kiselev_seq;

TEST(kiselev_i_shell_simple_seq, check_8_size) {
  std::vector<int> arr = {5, 3, 8, 6, 2, 7, 1, 4};
  std::vector<int> res(arr.size(), 0);
  std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  KiselevTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected, res);
}

TEST(kiselev_i_shell_simple_seq, check_100_size_reverse) {
  const int count = 100;

  // Create data
  std::vector<int> in(count, 0);
  std::vector<int> out(count, 0);
  std::vector<int> res(count, 0);
  for (int i = 0; i < count; i++) {
    in[i] = count - 1 - i;
    res[i] = i;
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  KiselevTaskSequential kiselevTaskSequential(taskDataSeq);
  ASSERT_EQ(kiselevTaskSequential.validation(), true);
  ASSERT_EQ(kiselevTaskSequential.pre_processing(), true);
  ASSERT_EQ(kiselevTaskSequential.run(), true);
  ASSERT_EQ(kiselevTaskSequential.post_processing(), true);
  ASSERT_EQ(out, res);
}

TEST(kiselev_i_shell_simple_seq, check_100_size_sorted) {
  const int count = 100;

  // Create data
  std::vector<int> in(count, 0);
  std::vector<int> out(count, 0);
  std::vector<int> res(count, 0);
  for (int i = 0; i < count; i++) {
    in[i] = i;
    res[i] = i;
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  KiselevTaskSequential kiselevTaskSequential(taskDataSeq);
  ASSERT_EQ(kiselevTaskSequential.validation(), true);
  ASSERT_EQ(kiselevTaskSequential.pre_processing(), true);
  ASSERT_EQ(kiselevTaskSequential.run(), true);
  ASSERT_EQ(kiselevTaskSequential.post_processing(), true);
  ASSERT_EQ(out, res);
}

TEST(kiselev_i_shell_simple_seq, check_100_size_bias) {
  const int count = 100;
  const int bias = 50;

  // Create data
  std::vector<int> in(count, 0);
  std::vector<int> out(count, 0);
  std::vector<int> res(count, 0);
  for (int i = 0; i < count; i++) {
    in[i] = count - 1 - i + bias;
    res[i] = i + bias;
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  KiselevTaskSequential kiselevTaskSequential(taskDataSeq);
  ASSERT_EQ(kiselevTaskSequential.validation(), true);
  ASSERT_EQ(kiselevTaskSequential.pre_processing(), true);
  ASSERT_EQ(kiselevTaskSequential.run(), true);
  ASSERT_EQ(kiselevTaskSequential.post_processing(), true);
  ASSERT_EQ(out, res);
}

TEST(kiselev_i_shell_simple_seq, check_incorrect_input) {
  const int count = 4;

  std::vector<std::vector<int>> in = {{2, 1}, {4, 3}};
  std::vector<int> out(count, 0);
  std::vector<int> res = {1, 2, 3, 4};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  KiselevTaskSequential kiselevTaskSequential(taskDataSeq);
  ASSERT_EQ(kiselevTaskSequential.validation(), false);
}
