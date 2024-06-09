// Copyright 2024 Bozin Dmitry
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "seq/bozin_d_trapez/include/ops_seq.hpp"

TEST(bozin_d_trapez_seq, Test1) {
  double ax = 0.0;
  double bx = 0.0;
  double ay = 0.0;
  double by = 0.0;

  int nx = 20;
  int ny = 20;

  double res = 0;

  auto sqr_x_y = [](double x, double y) { return sqrt(x * x + y * y); };

  // Create data
  std::vector<double> out(1, res);
  std::vector<double> in = {ax, ay, bx, by};
  std::vector<int> n = {nx, ny};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  BozinTaskSequential testTaskSequential(taskDataSeq, sqr_x_y);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 0.02);
}

TEST(bozin_d_trapez_seq, Test2) {
  double ax = 0.0;
  double bx = 1.0;
  double ay = 0.0;
  double by = 1.0;

  int nx = 10;
  int ny = 10;

  double res = 0.25;

  auto mul_x_y = [](double x, double y) { return x * y; };

  // Create data
  std::vector<double> out(1, res);
  std::vector<double> in = {ax, ay, bx, by};
  std::vector<int> n = {nx, ny};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  BozinTaskSequential testTaskSequential(taskDataSeq, mul_x_y);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 0.02);
}

TEST(bozin_d_trapez_seq, Test3) {
  double ax = 0.0;
  double bx = 3.14159265358979323846;
  double ay = 0.0;
  double by = 3.14159265358979323846;

  int nx = 100;
  int ny = 100;

  double res = 2 * 3.14159265358979323846;

  auto trigm_x_y = [](double x, double y) { return cos(x) + sin(y); };

  // Create data
  std::vector<double> out(1, res);
  std::vector<double> in = {ax, ay, bx, by};
  std::vector<int> n = {nx, ny};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  BozinTaskSequential testTaskSequential(taskDataSeq, trigm_x_y);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 0.02);
}

TEST(bozin_d_trapez_seq, Test4) {
  const double ax = 0.0;
  double bx = 3.14159265358979323846;
  double ay = 0.0;
  double by = 1.0;

  int nx = 50;
  int ny = 50;

  double res = 2.2685;

  auto trigmul_x_y = [](double x, double y) { return x * sin(y); };

  // Create data
  std::vector<double> out(1, res);
  std::vector<double> in = {ax, ay, bx, by};
  std::vector<int> n = {nx, ny};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  BozinTaskSequential testTaskSequential(taskDataSeq, trigmul_x_y);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 0.02);
}

TEST(bozin_d_trapez_seq, Test5) {
  double ax = 1.0;
  double bx = 2.0;
  double ay = 1.0;
  double by = 2.0;

  int nx = 300;
  int ny = 300;

  double res = 1.039;

  auto divide_x_y = [](double x, double y) { return x / y; };

  // Create data
  std::vector<double> out(1, res);
  std::vector<double> in = {ax, ay, bx, by};
  std::vector<int> n = {nx, ny};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  BozinTaskSequential testTaskSequential(taskDataSeq, divide_x_y);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 0.02);
}
