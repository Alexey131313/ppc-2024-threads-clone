// Copyright 2024 Korablev Nikita
#include <gtest/gtest.h>

#include <vector>

#include "omp/korablev_n_montecarlo/include/ops_omp.hpp"
#include "omp/korablev_n_montecarlo/include/ops_seq.hpp"
#define ESTIMATE 0.2

TEST(korablev_n_monte_carlo_omp, Test_lin_fun) {
  double res = 8;
  korablev_func f = korablev_flin;

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
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(in2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  // Create Task
  KorablevSequentialMonteCarlo testTaskSeq(taskDataSeq);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataOmp->inputs_count.emplace_back(in1.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataOmp->inputs_count.emplace_back(in2.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataOmp->inputs_count.emplace_back(in2.size());

  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataOmp->outputs_count.emplace_back(out_omp.size());

  KorablevOMPMonteCarlo testTaskOmp(taskDataOmp);
  ASSERT_EQ(testTaskOmp.validation(), true);
  testTaskOmp.pre_processing();
  testTaskOmp.run();
  testTaskOmp.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(korablev_n_monte_carlo_omp, Test_sum_of_sin) {
  double res = 5.67369;
  korablev_func f = korablev_fsinxPsiny;

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
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(in2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  // Create Task
  KorablevSequentialMonteCarlo testTaskSeq(taskDataSeq);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataOmp->inputs_count.emplace_back(in1.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataOmp->inputs_count.emplace_back(in2.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataOmp->inputs_count.emplace_back(in2.size());

  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataOmp->outputs_count.emplace_back(out_omp.size());

  KorablevOMPMonteCarlo testTaskOmp(taskDataOmp);
  ASSERT_EQ(testTaskOmp.validation(), true);
  testTaskOmp.pre_processing();
  testTaskOmp.run();
  testTaskOmp.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(korablev_n_monte_carlo_omp, Test_sum_of_cos) {
  double res = 6.22943;
  korablev_func f = korablev_fcosxPcosy;

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
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(in2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  // Create Task
  KorablevSequentialMonteCarlo testTaskSeq(taskDataSeq);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataOmp->inputs_count.emplace_back(in1.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataOmp->inputs_count.emplace_back(in2.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataOmp->inputs_count.emplace_back(in2.size());

  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataOmp->outputs_count.emplace_back(out_omp.size());

  KorablevOMPMonteCarlo testTaskOmp(taskDataOmp);
  ASSERT_EQ(testTaskOmp.validation(), true);
  testTaskOmp.pre_processing();
  testTaskOmp.run();
  testTaskOmp.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(korablev_n_monte_carlo_omp, Test_x_mult_y) {
  double res = 2.25;
  korablev_func f = korablev_fxy;

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
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(in2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  // Create Task
  KorablevSequentialMonteCarlo testTaskSeq(taskDataSeq);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataOmp->inputs_count.emplace_back(in1.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataOmp->inputs_count.emplace_back(in2.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataOmp->inputs_count.emplace_back(in2.size());

  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataOmp->outputs_count.emplace_back(out_omp.size());

  KorablevOMPMonteCarlo testTaskOmp(taskDataOmp);
  ASSERT_EQ(testTaskOmp.validation(), true);
  testTaskOmp.pre_processing();
  testTaskOmp.run();
  testTaskOmp.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(korablev_n_monte_carlo_omp, Test_x_mult_y_mult_y) {
  double res = 1.5;
  korablev_func f = korablev_fxyy;

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
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(in2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  // Create Task
  KorablevSequentialMonteCarlo testTaskSeq(taskDataSeq);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataOmp->inputs_count.emplace_back(in1.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataOmp->inputs_count.emplace_back(in2.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataOmp->inputs_count.emplace_back(in2.size());

  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataOmp->outputs_count.emplace_back(out_omp.size());

  KorablevOMPMonteCarlo testTaskOmp(taskDataOmp);
  ASSERT_EQ(testTaskOmp.validation(), true);
  testTaskOmp.pre_processing();
  testTaskOmp.run();
  testTaskOmp.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}