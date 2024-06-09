// copyright 2024 Travin Maksim
#include <gtest/gtest.h>

#include <omp/travin_m_simpson_m/include/ops_omp.hpp>
#include <vector>
using namespace travin_omp;

TEST(travin_m_simpson_omp, test_lin) {
  int a = 6;
  int b = 10;
  int c = 6;
  int d = 10;
  int n = 100;
  double res = 128;
  function func = LinFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> seq{res};
  std::vector<double> omp{res};

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

  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataOmp->inputs_count.emplace_back(in.size());
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(omp.data()));
  taskDataOmp->outputs_count.emplace_back(omp.size());

  SimpsonTaskOmp TaskOmp(taskDataOmp, func);
  ASSERT_EQ(TaskOmp.validation(), true);
  TaskOmp.pre_processing();
  TaskOmp.run();
  TaskOmp.post_processing();
  ASSERT_NEAR(omp[0], seq[0], 0.001);
}

TEST(travin_m_simpson_omp, test_trig) {
  int a = 6;
  int b = 10;
  int c = 6;
  int d = 10;
  int n = 100;
  double res = -3.16725;
  function func = TrigFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> seq{res};
  std::vector<double> omp{res};

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

  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataOmp->inputs_count.emplace_back(in.size());
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(omp.data()));
  taskDataOmp->outputs_count.emplace_back(omp.size());

  SimpsonTaskOmp TaskOmp(taskDataOmp, func);
  ASSERT_EQ(TaskOmp.validation(), true);
  TaskOmp.pre_processing();
  TaskOmp.run();
  TaskOmp.post_processing();
  ASSERT_NEAR(omp[0], seq[0], 0.001);
}

TEST(travin_m_simpson_omp, test_mul) {
  int a = 6;
  int b = 10;
  int c = 6;
  int d = 10;
  int n = 100;
  double res = 1024;
  function func = MulFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> seq{res};
  std::vector<double> omp{res};

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

  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataOmp->inputs_count.emplace_back(in.size());
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(omp.data()));
  taskDataOmp->outputs_count.emplace_back(omp.size());

  SimpsonTaskOmp TaskOmp(taskDataOmp, func);
  ASSERT_EQ(TaskOmp.validation(), true);
  TaskOmp.pre_processing();
  TaskOmp.run();
  TaskOmp.post_processing();
  ASSERT_NEAR(omp[0], seq[0], 0.001);
}

TEST(travin_m_simpson_omp, test_exp) {
  int a = 0;
  int b = 2;
  int c = 0;
  int d = 2;
  int n = 100;
  double res = 40.82;
  function func = ExpFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> seq{res};
  std::vector<double> omp{res};

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

  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataOmp->inputs_count.emplace_back(in.size());
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(omp.data()));
  taskDataOmp->outputs_count.emplace_back(omp.size());

  SimpsonTaskOmp TaskOmp(taskDataOmp, func);
  ASSERT_EQ(TaskOmp.validation(), true);
  TaskOmp.pre_processing();
  TaskOmp.run();
  TaskOmp.post_processing();
  ASSERT_NEAR(omp[0], seq[0], 0.001);
}

TEST(travin_m_simpson_omp, test_rand) {
  int a = 3;
  int b = 6;
  int c = 3;
  int d = 6;
  int n = 100;
  double res = 378;
  function func = RandFunc;

  std::vector<int> in{a, b, c, d, n};
  std::vector<double> seq{res};
  std::vector<double> omp{res};

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

  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataOmp->inputs_count.emplace_back(in.size());
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(omp.data()));
  taskDataOmp->outputs_count.emplace_back(omp.size());

  SimpsonTaskOmp TaskOmp(taskDataOmp, func);
  ASSERT_EQ(TaskOmp.validation(), true);
  TaskOmp.pre_processing();
  TaskOmp.run();
  TaskOmp.post_processing();
  ASSERT_NEAR(omp[0], seq[0], 0.001);
}