// copyright 2024 Travin Maksim
#include <gtest/gtest.h>

#include <tbb/travin_m_simpson_m/include/ops_tbb.hpp>
#include <vector>
using namespace travin_tbb;

TEST(travin_m_simpson_tbb, test_lin) {
  int a = 6;
  int b = 10;
  int c = 6;
  int d = 10;
  int n = 100;
  double res = 128;
  function func = LinFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> seq{res};
  std::vector<double> tbb{res};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq.data()));
  taskDataSeq->outputs_count.emplace_back(seq.size());

  SimpsonTaskSeq TaskSeq(taskDataSeq, func);
  ASSERT_EQ(TaskSeq.validation(), true);
  TaskSeq.pre_processing();
  TaskSeq.run();
  TaskSeq.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataTbb->inputs_count.emplace_back(in.size());
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(tbb.data()));
  taskDataTbb->outputs_count.emplace_back(tbb.size());

  SimpsonTaskTbb TaskOmp(taskDataTbb, func);
  ASSERT_EQ(TaskOmp.validation(), true);
  TaskOmp.pre_processing();
  TaskOmp.run();
  TaskOmp.post_processing();
  ASSERT_NEAR(tbb[0], seq[0], 0.001);
}

TEST(travin_m_simpson_tbb, test_trig) {
  int a = 6;
  int b = 10;
  int c = 6;
  int d = 10;
  int n = 100;
  double res = -3.16725;
  function func = TrigFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> seq{res};
  std::vector<double> tbb{res};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq.data()));
  taskDataSeq->outputs_count.emplace_back(seq.size());

  SimpsonTaskSeq TaskSeq(taskDataSeq, func);
  ASSERT_EQ(TaskSeq.validation(), true);
  TaskSeq.pre_processing();
  TaskSeq.run();
  TaskSeq.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataTbb->inputs_count.emplace_back(in.size());
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(tbb.data()));
  taskDataTbb->outputs_count.emplace_back(tbb.size());

  SimpsonTaskTbb TaskOmp(taskDataTbb, func);
  ASSERT_EQ(TaskOmp.validation(), true);
  TaskOmp.pre_processing();
  TaskOmp.run();
  TaskOmp.post_processing();
  ASSERT_NEAR(tbb[0], seq[0], 0.001);
}

TEST(travin_m_simpson_tbb, test_mul) {
  int a = 6;
  int b = 10;
  int c = 6;
  int d = 10;
  int n = 100;
  double res = 1024;
  function func = MulFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> seq{res};
  std::vector<double> tbb{res};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq.data()));
  taskDataSeq->outputs_count.emplace_back(seq.size());

  SimpsonTaskSeq TaskSeq(taskDataSeq, func);
  ASSERT_EQ(TaskSeq.validation(), true);
  TaskSeq.pre_processing();
  TaskSeq.run();
  TaskSeq.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataTbb->inputs_count.emplace_back(in.size());
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(tbb.data()));
  taskDataTbb->outputs_count.emplace_back(tbb.size());

  SimpsonTaskTbb TaskOmp(taskDataTbb, func);
  ASSERT_EQ(TaskOmp.validation(), true);
  TaskOmp.pre_processing();
  TaskOmp.run();
  TaskOmp.post_processing();
  ASSERT_NEAR(tbb[0], seq[0], 0.001);
}

TEST(travin_m_simpson_tbb, test_exp) {
  int a = 0;
  int b = 2;
  int c = 0;
  int d = 2;
  int n = 100;
  double res = 40.82;
  function func = ExpFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> seq{res};
  std::vector<double> tbb{res};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq.data()));
  taskDataSeq->outputs_count.emplace_back(seq.size());

  SimpsonTaskSeq TaskSeq(taskDataSeq, func);
  ASSERT_EQ(TaskSeq.validation(), true);
  TaskSeq.pre_processing();
  TaskSeq.run();
  TaskSeq.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataTbb->inputs_count.emplace_back(in.size());
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(tbb.data()));
  taskDataTbb->outputs_count.emplace_back(tbb.size());

  SimpsonTaskTbb TaskOmp(taskDataTbb, func);
  ASSERT_EQ(TaskOmp.validation(), true);
  TaskOmp.pre_processing();
  TaskOmp.run();
  TaskOmp.post_processing();
  ASSERT_NEAR(tbb[0], seq[0], 0.001);
}

TEST(travin_m_simpson_tbb, test_rand) {
  int a = 3;
  int b = 6;
  int c = 3;
  int d = 6;
  int n = 100;
  double res = 378;
  function func = RandFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> seq{res};
  std::vector<double> tbb{res};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq.data()));
  taskDataSeq->outputs_count.emplace_back(seq.size());

  SimpsonTaskSeq TaskSeq(taskDataSeq, func);
  ASSERT_EQ(TaskSeq.validation(), true);
  TaskSeq.pre_processing();
  TaskSeq.run();
  TaskSeq.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataTbb->inputs_count.emplace_back(in.size());
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(tbb.data()));
  taskDataTbb->outputs_count.emplace_back(tbb.size());

  SimpsonTaskTbb TaskOmp(taskDataTbb, func);
  ASSERT_EQ(TaskOmp.validation(), true);
  TaskOmp.pre_processing();
  TaskOmp.run();
  TaskOmp.post_processing();
  ASSERT_NEAR(tbb[0], seq[0], 0.001);
}