// Copyright 2024 Andrey Derun
#include <gtest/gtest.h>

#include <vector>

#include "omp/derun_a_shell/include/shell_omp.hpp"

void runTestRandom(int count) {
  // Create data
  std::vector<int> in = ShellOMP::generate_random_vector(count, 1, 100);
  std::vector<int> out(count, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataOMP = std::make_shared<ppc::core::TaskData>();  // NOLINT
  taskDataOMP->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataOMP->inputs_count.emplace_back(in.size());
  taskDataOMP->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataOMP->outputs_count.emplace_back(out.size());

  // Create Task
  ShellOMP testTaskOMP(taskDataOMP);
  ASSERT_EQ(testTaskOMP.validation(), true);
  testTaskOMP.pre_processing();
  testTaskOMP.run();
  testTaskOMP.post_processing();
  ASSERT_TRUE(ShellOMP::checkSorted(out));
}

TEST(derun_andrey_shell_omp, Shell_Validation_Fail) {
  const int count = 10;

  // Create data
  std::vector<int> in1 = ShellOMP::generate_random_vector(count, 1, 100);
  std::vector<int> in2 = std::vector<int>(5);
  std::vector<int> out(count, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataOMP = std::make_shared<ppc::core::TaskData>();  // NOLINT
  taskDataOMP->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataOMP->inputs_count.emplace_back(in1.size());

  taskDataOMP->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataOMP->inputs_count.emplace_back(in2.size());

  taskDataOMP->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataOMP->outputs_count.emplace_back(out.size());

  // Create Task
  ShellOMP testTaskOMP(taskDataOMP);
  ASSERT_EQ(testTaskOMP.validation(), false);
}

TEST(derun_andrey_shell_omp, Shell_Random_10) {
  const int count = 10;

  runTestRandom(count);
}

TEST(derun_andrey_shell_omp, Shell_Random_20) {
  const int count = 20;

  runTestRandom(count);
}

TEST(derun_andrey_shell_omp, Shell_Random_50) {
  const int count = 50;

  runTestRandom(count);
}

TEST(derun_andrey_shell_omp, Shell_Random_70) {
  const int count = 70;

  runTestRandom(count);
}

TEST(derun_andrey_shell_omp, Shell_Random_100) {
  const int count = 100;
  runTestRandom(count);
}
