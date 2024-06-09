// Copyright 2024 Andrey Derun
#include <gtest/gtest.h>

#include <vector>

#include "seq/derun_a_shell/include/shell_seq.hpp"

void runTestRandom(int count) {
  // Create data
  std::vector<int> in = ShellSequential::generate_random_vector(count, 1, 100);
  std::vector<int> out(count, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();  // NOLINT
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ShellSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_TRUE(ShellSequential::checkSorted(out));
}

TEST(derun_andrey_shell_seq, Shell_Validation_Fail) {
  const int count = 10;

  // Create data
  std::vector<int> in1 = ShellSequential::generate_random_vector(count, 1, 100);
  std::vector<int> in2 = std::vector<int>(5);
  std::vector<int> out(count, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();  // NOLINT
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ShellSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(derun_andrey_shell_seq, Shell_Random_10) {
  const int count = 10;

  runTestRandom(count);
}

TEST(derun_andrey_shell_seq, Shell_Random_20) {
  const int count = 20;

  runTestRandom(count);
}

TEST(derun_andrey_shell_seq, Shell_Random_50) {
  const int count = 50;

  runTestRandom(count);
}

TEST(derun_andrey_shell_seq, Shell_Random_70) {
  const int count = 70;

  runTestRandom(count);
}

TEST(derun_andrey_shell_seq, Shell_Random_100) {
  const int count = 100;
  runTestRandom(count);
}
