// Copyright 2024 Korablev Nikita
#include <gtest/gtest.h>

#include <vector>

#include "tbb/korablev_n_montecarlo/include/ops_seq.hpp"
#include "tbb/korablev_n_montecarlo/include/ops_tbb.hpp"
#define ESTIMATE 0.2

TEST(Korablev_N_Monte_Carlo_TBB, Test_lin_fun) {
  double res = 8;
  func f = flin;

  // Create data
  std::vector<double> in1 = {0, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out_seq(1, res);
  std::vector<double> out_omp(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  // Create Task
  KorablevSequentialMonteCarlo testTaskSeq(taskDataSeq, f);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataTBB->inputs_count.emplace_back(in1.size());
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataTBB->inputs_count.emplace_back(in2.size());

  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataTBB->outputs_count.emplace_back(out_omp.size());

  KorablevMonteCarloTBB testTaskTBB(taskDataTBB, f);
  ASSERT_EQ(testTaskTBB.validation(), true);
  testTaskTBB.pre_processing();
  testTaskTBB.run();
  testTaskTBB.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(Korablev_N_Monte_Carlo_TBB, Test_sum_of_sin) {
  double res = 5.67369;
  func f = fsinxPsiny;

  // Create data
  std::vector<double> in1 = {-1, 2};
  std::vector<double> in2 = {-1, 2};
  std::vector<double> out_seq(1, res);
  std::vector<double> out_omp(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  // Create Task
  KorablevSequentialMonteCarlo testTaskSeq(taskDataSeq, f);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataTBB->inputs_count.emplace_back(in1.size());
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataTBB->inputs_count.emplace_back(in2.size());

  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataTBB->outputs_count.emplace_back(out_omp.size());

  KorablevMonteCarloTBB testTaskTBB(taskDataTBB, f);
  ASSERT_EQ(testTaskTBB.validation(), true);
  testTaskTBB.pre_processing();
  testTaskTBB.run();
  testTaskTBB.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(Korablev_N_Monte_Carlo_TBB, Test_sum_of_cos) {
  double res = 6.22943;
  func f = fcosxPcosy;

  // Create data
  std::vector<double> in1 = {-1, 2};
  std::vector<double> in2 = {0, 2};
  std::vector<double> out_seq(1, res);
  std::vector<double> out_omp(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  // Create Task
  KorablevSequentialMonteCarlo testTaskSeq(taskDataSeq, f);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataTBB->inputs_count.emplace_back(in1.size());
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataTBB->inputs_count.emplace_back(in2.size());

  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataTBB->outputs_count.emplace_back(out_omp.size());

  KorablevMonteCarloTBB testTaskTBB(taskDataTBB, f);
  ASSERT_EQ(testTaskTBB.validation(), true);
  testTaskTBB.pre_processing();
  testTaskTBB.run();
  testTaskTBB.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(Korablev_N_Monte_Carlo_TBB, Test_x_mult_y) {
  double res = 2.25;
  func f = fxy;

  // Create data
  std::vector<double> in1 = {0, 1};
  std::vector<double> in2 = {0, 3};
  std::vector<double> out_seq(1, res);
  std::vector<double> out_omp(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  // Create Task
  KorablevSequentialMonteCarlo testTaskSeq(taskDataSeq, f);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataTBB->inputs_count.emplace_back(in1.size());
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataTBB->inputs_count.emplace_back(in2.size());

  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataTBB->outputs_count.emplace_back(out_omp.size());

  KorablevMonteCarloTBB testTaskTBB(taskDataTBB, f);
  ASSERT_EQ(testTaskTBB.validation(), true);
  testTaskTBB.pre_processing();
  testTaskTBB.run();
  testTaskTBB.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(Korablev_N_Monte_Carlo_TBB, Test_x_mult_y_mult_y) {
  double res = 1.5;
  func f = fxyy;

  // Create data
  std::vector<double> in1 = {0, 3};
  std::vector<double> in2 = {0, 1};
  std::vector<double> out_seq(1, res);
  std::vector<double> out_omp(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  // Create Task
  KorablevSequentialMonteCarlo testTaskSeq(taskDataSeq, f);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataTBB->inputs_count.emplace_back(in1.size());
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataTBB->inputs_count.emplace_back(in2.size());

  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataTBB->outputs_count.emplace_back(out_omp.size());

  KorablevMonteCarloTBB testTaskTBB(taskDataTBB, f);
  ASSERT_EQ(testTaskTBB.validation(), true);
  testTaskTBB.pre_processing();
  testTaskTBB.run();
  testTaskTBB.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}