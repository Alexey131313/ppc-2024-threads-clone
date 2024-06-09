// Copyright 2024 Vinokurov Ivan
#include <gtest/gtest.h>

#include <vector>

#include "seq/vinokurov_i_simpson_method/include/ops_seq.hpp"

TEST(vinokurov_i_simpson_method_seq, test_linear) {
  const int a = 0;
  const int b = 10;
  const int c = 0;
  const int d = 10;
  const int n = 100;

  const double result = 90;

  func fn = fn_linear;

  // Create data

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> out(1, result);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> dataSeq = std::make_shared<ppc::core::TaskData>();
  dataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  dataSeq->inputs_count.emplace_back(in.size());
  dataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  dataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  SimpsonMethodSequential testTaskSequential(dataSeq, fn);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_LE(result - out[0], 0.01);
}

TEST(vinokurov_i_simpson_method_seq, test_exp) {
  const int a = 0;
  const int b = 10;
  const int c = 0;
  const int d = 10;
  const int n = 100;

  double result = 27.18;

  func fn = fn_exp;

  // Create data
  std::vector<int> in{a, b, c, d, n};
  std::vector<double> out(1, result);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  SimpsonMethodSequential testTaskSequential(taskDataSeq, fn);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_LE(result - out[0], 0.01);
}

TEST(vinokurov_i_simpson_method_seq, test_trig) {
  const int a = 6;
  const int b = 18;
  const int c = 6;
  const int d = 18;
  const int n = 100;

  double result = 0.04;

  func fn = fn_trig;

  // Create data
  std::vector<int> in{a, b, c, d, n};
  std::vector<double> out(1, result);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  SimpsonMethodSequential testTaskSequential(taskDataSeq, fn);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_LE(result - out[0], 0.01);
}

TEST(vinokurov_i_simpson_method_seq, test_mul) {
  const int a = 0;
  const int b = 10;
  const int c = 0;
  const int d = 10;
  const int n = 100;

  double result = 16666.66;

  func fn = fn_mul;

  // Create data
  std::vector<int> in{a, b, c, d, n};
  std::vector<double> out(1, result);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  SimpsonMethodSequential testTaskSequential(taskDataSeq, fn);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_LE(result - out[0], 0.01);
}

TEST(vinokurov_i_simpson_method_seq, test_other) {
  const int a = 0;
  const int b = 10;
  const int c = 0;
  const int d = 10;
  const int n = 50;

  double result = 5833.33;

  func fn = fn_other;

  // Create data
  std::vector<int> in{a, b, c, d, n};
  std::vector<double> out(1, result);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  SimpsonMethodSequential testTaskSequential(taskDataSeq, fn);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_LE(result - out[0], 0.01);
}
