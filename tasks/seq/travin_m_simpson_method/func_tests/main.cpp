// copyright 2024 Travin Maksim
#include <gtest/gtest.h>

#include <seq/travin_m_simpson_method/include/ops_seq.hpp>
#include <vector>

TEST(travin_m_simpson, test_lin) {
  int a = 6;
  int b = 10;
  int c = 6;
  int d = 10;
  int n = 100;
  double res = 128;
  function func = LinFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> out{res};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  SimpsonTask TaskSeq(taskDataSeq, func);
  ASSERT_EQ(TaskSeq.validation(), true);
  TaskSeq.pre_processing();
  TaskSeq.run();
  TaskSeq.post_processing();
  ASSERT_NEAR(res, out[0], 0.001);
}

TEST(travin_m_simpson, test_trig) {
  int a = 6;
  int b = 10;
  int c = 6;
  int d = 10;
  int n = 100;
  double res = -3.16725;
  function func = TrigFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> out{res};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  SimpsonTask TaskSeq(taskDataSeq, func);
  ASSERT_EQ(TaskSeq.validation(), true);
  TaskSeq.pre_processing();
  TaskSeq.run();
  TaskSeq.post_processing();
  ASSERT_NEAR(res, out[0], 0.001);
}

TEST(travin_m_simpson, test_mul) {
  int a = 6;
  int b = 10;
  int c = 6;
  int d = 10;
  int n = 100;
  double res = 1024;
  function func = MulFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> out{res};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  SimpsonTask TaskSeq(taskDataSeq, func);
  ASSERT_EQ(TaskSeq.validation(), true);
  TaskSeq.pre_processing();
  TaskSeq.run();
  TaskSeq.post_processing();
  ASSERT_NEAR(res, out[0], 0.001);
}

TEST(travin_m_simpson, test_exp) {
  int a = 0;
  int b = 2;
  int c = 0;
  int d = 2;
  int n = 100;
  double res = 40.82;
  function func = ExpFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> out{res};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  SimpsonTask TaskSeq(taskDataSeq, func);
  ASSERT_EQ(TaskSeq.validation(), true);
  TaskSeq.pre_processing();
  TaskSeq.run();
  TaskSeq.post_processing();
  ASSERT_NEAR(res, out[0], 0.001);
}

TEST(travin_m_simpson, test_rand) {
  int a = 3;
  int b = 6;
  int c = 3;
  int d = 6;
  int n = 100;
  double res = 378;
  function func = RandFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> out{res};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  SimpsonTask TaskSeq(taskDataSeq, func);
  ASSERT_EQ(TaskSeq.validation(), true);
  TaskSeq.pre_processing();
  TaskSeq.run();
  TaskSeq.post_processing();
  ASSERT_NEAR(res, out[0], 0.001);
}