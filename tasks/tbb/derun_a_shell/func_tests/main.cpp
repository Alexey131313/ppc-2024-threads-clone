// Copyright 2024 Andrey Derun
#include <gtest/gtest.h>

#include <vector>

#include "tbb/derun_a_shell/include/shell_tbb.hpp"

void runTestRandom(int count) {
  // Create data
  std::vector<int> in = ShellTBB::generate_random_vector(count, 1, 100);
  std::vector<int> out(count, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();  // NOLINT
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataTBB->inputs_count.emplace_back(in.size());
  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataTBB->outputs_count.emplace_back(out.size());

  // Create Task
  ShellTBB testTaskTBB(taskDataTBB);
  ASSERT_EQ(testTaskTBB.validation(), true);
  testTaskTBB.pre_processing();
  testTaskTBB.run();
  testTaskTBB.post_processing();
  ASSERT_TRUE(ShellTBB::checkSorted(out));
}

TEST(derun_andrey_shell_tbb, Shell_Validation_Fail) {
  const int count = 10;

  // Create data
  std::vector<int> in1 = ShellTBB::generate_random_vector(count, 1, 100);
  std::vector<int> in2 = std::vector<int>(5);
  std::vector<int> out(count, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();  // NOLINT
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataTBB->inputs_count.emplace_back(in1.size());

  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataTBB->inputs_count.emplace_back(in2.size());

  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataTBB->outputs_count.emplace_back(out.size());

  // Create Task
  ShellTBB testTaskTBB(taskDataTBB);
  ASSERT_EQ(testTaskTBB.validation(), false);
}

TEST(derun_andrey_shell_tbb, Shell_Random_10) {
  const int count = 10;

  runTestRandom(count);
}

TEST(derun_andrey_shell_tbb, Shell_Random_20) {
  const int count = 20;

  runTestRandom(count);
}

TEST(derun_andrey_shell_tbb, Shell_Random_50) {
  const int count = 50;

  runTestRandom(count);
}

TEST(derun_andrey_shell_tbb, Shell_Random_70) {
  const int count = 70;

  runTestRandom(count);
}

TEST(derun_andrey_shell_tbb, Shell_Random_100) {
  const int count = 100;
  runTestRandom(count);
}
