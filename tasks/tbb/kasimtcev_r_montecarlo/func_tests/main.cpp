// Copyright 2024 Kasimtcev Roman
#include <gtest/gtest.h>

#include <vector>

#include "tbb/kasimtcev_r_montecarlo/include/ops_seq.hpp"
#include "tbb/kasimtcev_r_montecarlo/include/ops_tbb.hpp"
#define ESTIMATE 0.2

TEST(kasimtcev_r_monteCarlo_TBB, Tested_linear) {
  double res = 8;
  func f = flinear;

  std::vector<double> a1 = {0, 2};
  std::vector<double> a2 = {0, 2};
  std::vector<double> out_seq(1, res);
  std::vector<double> out_omp(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataSeq->inputs_count.emplace_back(a1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataSeq->inputs_count.emplace_back(a2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  KasimtcevSequentialMonteCarlo testTaskSeq(taskDataSeq, f);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataTBB->inputs_count.emplace_back(a1.size());
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataTBB->inputs_count.emplace_back(a2.size());

  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataTBB->outputs_count.emplace_back(out_omp.size());

  KasimtcevMonteCarloTBB testTaskTBB(taskDataTBB, f);
  ASSERT_EQ(testTaskTBB.validation(), true);
  testTaskTBB.pre_processing();
  testTaskTBB.run();
  testTaskTBB.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(kasimtcev_r_monteCarlo_TBB, Tested_summ_sin) {
  double res = 5.67369;
  func f = fsinxsiny;

  std::vector<double> a1 = {-1, 2};
  std::vector<double> a2 = {-1, 2};
  std::vector<double> out_seq(1, res);
  std::vector<double> out_omp(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataSeq->inputs_count.emplace_back(a1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataSeq->inputs_count.emplace_back(a2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  KasimtcevSequentialMonteCarlo testTaskSeq(taskDataSeq, f);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataTBB->inputs_count.emplace_back(a1.size());
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataTBB->inputs_count.emplace_back(a2.size());

  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataTBB->outputs_count.emplace_back(out_omp.size());

  KasimtcevMonteCarloTBB testTaskTBB(taskDataTBB, f);
  ASSERT_EQ(testTaskTBB.validation(), true);
  testTaskTBB.pre_processing();
  testTaskTBB.run();
  testTaskTBB.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(kasimtcev_r_monteCarlo_TBB, Tested_sum_cos) {
  double res = 6.22943;
  func f = fcosxcosy;

  std::vector<double> a1 = {-1, 2};
  std::vector<double> a2 = {0, 2};
  std::vector<double> out_seq(1, res);
  std::vector<double> out_omp(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataSeq->inputs_count.emplace_back(a1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataSeq->inputs_count.emplace_back(a2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  KasimtcevSequentialMonteCarlo testTaskSeq(taskDataSeq, f);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataTBB->inputs_count.emplace_back(a1.size());
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataTBB->inputs_count.emplace_back(a2.size());

  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataTBB->outputs_count.emplace_back(out_omp.size());

  KasimtcevMonteCarloTBB testTaskTBB(taskDataTBB, f);
  ASSERT_EQ(testTaskTBB.validation(), true);
  testTaskTBB.pre_processing();
  testTaskTBB.run();
  testTaskTBB.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(kasimtcev_r_monteCarlo_TBB, Tested_x_multiply_y) {
  double res = 2.25;
  func f = fxy1;

  std::vector<double> a1 = {0, 1};
  std::vector<double> a2 = {0, 3};
  std::vector<double> out_seq(1, res);
  std::vector<double> out_omp(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataSeq->inputs_count.emplace_back(a1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataSeq->inputs_count.emplace_back(a2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  KasimtcevSequentialMonteCarlo testTaskSeq(taskDataSeq, f);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataTBB->inputs_count.emplace_back(a1.size());
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataTBB->inputs_count.emplace_back(a2.size());

  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataTBB->outputs_count.emplace_back(out_omp.size());

  KasimtcevMonteCarloTBB testTaskTBB(taskDataTBB, f);
  ASSERT_EQ(testTaskTBB.validation(), true);
  testTaskTBB.pre_processing();
  testTaskTBB.run();
  testTaskTBB.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(kasimtcev_r_monteCarlo_TBB, Tested_x_multiply_yy) {
  double res = 1.5;
  func f = fxyy1;

  std::vector<double> a1 = {0, 3};
  std::vector<double> a2 = {0, 1};
  std::vector<double> out_seq(1, res);
  std::vector<double> out_omp(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataSeq->inputs_count.emplace_back(a1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataSeq->inputs_count.emplace_back(a2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  KasimtcevSequentialMonteCarlo testTaskSeq(taskDataSeq, f);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataTBB->inputs_count.emplace_back(a1.size());
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataTBB->inputs_count.emplace_back(a2.size());

  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataTBB->outputs_count.emplace_back(out_omp.size());

  KasimtcevMonteCarloTBB testTaskTBB(taskDataTBB, f);
  ASSERT_EQ(testTaskTBB.validation(), true);
  testTaskTBB.pre_processing();
  testTaskTBB.run();
  testTaskTBB.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}
