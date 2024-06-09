// Copyright 2024 Ivanov Nikita
#include <gtest/gtest.h>

#include <vector>

#include "seq/ivanov_n_int_simpson/include/ops_seq.hpp"

TEST(ivanov_n_int_simpson_seq, Linar_0_2) {
  const int a = 0;
  const int b = 2;
  const int c = 0;
  const int d = 2;
  const int n = 100;
  const int res = 8;
  func function = linear_fun;

  // Create data
  std::vector<int> in = {a, b, c, d, n};
  std::vector<double> out(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TestTaskSequentialSimpson testTaskSequential(taskDataSeq, function);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_LE(res - out[0], 0.001);
}

TEST(ivanov_n_int_simpson_seq, x_mul_y_5_10) {
  const int a = 5;
  const int b = 10;
  const int c = 2;
  const int d = 10;
  const int n = 100;
  const double res = 1406.25;
  func function = x_mul_y;

  // Create data
  std::vector<int> in = {a, b, c, d, n};
  std::vector<double> out(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TestTaskSequentialSimpson testTaskSequential(taskDataSeq, function);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_LE(res - out[0], 0.001);
}

TEST(ivanov_n_int_simpson_seq, sin_cos_0_2_0_1) {
  const int a = 0;
  const int b = 2;
  const int c = 0;
  const int d = 1;
  const int n = 100;
  const double res = 1.828692;
  func function = sin_cos;

  // Create data
  std::vector<int> in = {a, b, c, d, n};
  std::vector<double> out(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TestTaskSequentialSimpson testTaskSequential(taskDataSeq, function);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_LE(res - out[0], 0.001);
}

TEST(ivanov_n_int_simpson_seq, liner_10_12_2_4) {
  const int a = 10;
  const int b = 12;
  const int c = 2;
  const int d = 4;
  const int n = 100;
  const double res = 56.0;
  func function = linear_fun;

  // Create data
  std::vector<int> in = {a, b, c, d, n};
  std::vector<double> out(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TestTaskSequentialSimpson testTaskSequential(taskDataSeq, function);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_LE(res - out[0], 0.001);
}

TEST(ivanov_n_int_simpson_seq, Test_Sum_100) {
  const int a = 10;
  const int b = 12;
  const int c = 2;
  const int d = 4;
  const int n = 100;
  const double res = 132.0;
  func function = x_mul_y;

  // Create data
  std::vector<int> in = {a, b, c, d, n};
  std::vector<double> out(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TestTaskSequentialSimpson testTaskSequential(taskDataSeq, function);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_LE(res - out[0], 0.001);
}
