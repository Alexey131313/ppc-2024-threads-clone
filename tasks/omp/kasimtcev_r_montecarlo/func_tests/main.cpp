// Copyright 2024 Kasimtcev Roman
#include <gtest/gtest.h>

#include <vector>

#include "omp/kasimtcev_r_montecarlo/include/ops_omp.hpp"
#include "omp/kasimtcev_r_montecarlo/include/ops_seq.hpp"
#define ESTIMATE 0.2

TEST(kasimtcev_r_monte_carlo_omp, Tested_Linear) {
  double res = 8;
  kasimtcev_func f = kasimtcev_flinear;

  std::vector<double> a1 = {0, 2};
  std::vector<double> a2 = {0, 2};
  std::vector<double> out_seq(1, res);
  std::vector<double> out_omp(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataSeq->inputs_count.emplace_back(a1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataSeq->inputs_count.emplace_back(a2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(a2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  KasimtcevSequentialMonteCarlo testTaskSeq(taskDataSeq);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataOmp->inputs_count.emplace_back(a1.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataOmp->inputs_count.emplace_back(a2.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataOmp->inputs_count.emplace_back(a2.size());

  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataOmp->outputs_count.emplace_back(out_omp.size());

  KasimtcevOMPMonteCarlo testTaskOmp(taskDataOmp);
  ASSERT_EQ(testTaskOmp.validation(), true);
  testTaskOmp.pre_processing();
  testTaskOmp.run();
  testTaskOmp.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(kasimtcev_r_monte_carlo_omp, Tested_summ_sin) {
  double res = 5.67369;
  kasimtcev_func f = kasimtcev_fsinxsiny;

  std::vector<double> a1 = {-1, 2};
  std::vector<double> a2 = {-1, 2};
  std::vector<double> out_seq(1, res);
  std::vector<double> out_omp(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataSeq->inputs_count.emplace_back(a1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataSeq->inputs_count.emplace_back(a2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(a2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  KasimtcevSequentialMonteCarlo testTaskSeq(taskDataSeq);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataOmp->inputs_count.emplace_back(a1.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataOmp->inputs_count.emplace_back(a2.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataOmp->inputs_count.emplace_back(a2.size());

  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataOmp->outputs_count.emplace_back(out_omp.size());

  KasimtcevOMPMonteCarlo testTaskOmp(taskDataOmp);
  ASSERT_EQ(testTaskOmp.validation(), true);
  testTaskOmp.pre_processing();
  testTaskOmp.run();
  testTaskOmp.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(kasimtcev_r_monte_carlo_omp, Tested_summ_cos) {
  double res = 6.22943;
  kasimtcev_func f = kasimtcev_fcosxcosy;

  std::vector<double> a1 = {-1, 2};
  std::vector<double> a2 = {0, 2};
  std::vector<double> out_seq(1, res);
  std::vector<double> out_omp(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataSeq->inputs_count.emplace_back(a1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataSeq->inputs_count.emplace_back(a2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(a2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  KasimtcevSequentialMonteCarlo testTaskSeq(taskDataSeq);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataOmp->inputs_count.emplace_back(a1.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataOmp->inputs_count.emplace_back(a2.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataOmp->inputs_count.emplace_back(a2.size());

  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataOmp->outputs_count.emplace_back(out_omp.size());

  KasimtcevOMPMonteCarlo testTaskOmp(taskDataOmp);
  ASSERT_EQ(testTaskOmp.validation(), true);
  testTaskOmp.pre_processing();
  testTaskOmp.run();
  testTaskOmp.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(kasimtcev_r_monte_carlo_omp, Tested_x_multiply_y) {
  double res = 2.25;
  kasimtcev_func f = kasimtcev_fxy;

  std::vector<double> a1 = {0, 1};
  std::vector<double> a2 = {0, 3};
  std::vector<double> out_seq(1, res);
  std::vector<double> out_omp(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataSeq->inputs_count.emplace_back(a1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataSeq->inputs_count.emplace_back(a2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(a2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  KasimtcevSequentialMonteCarlo testTaskSeq(taskDataSeq);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataOmp->inputs_count.emplace_back(a1.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataOmp->inputs_count.emplace_back(a2.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataOmp->inputs_count.emplace_back(a2.size());

  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataOmp->outputs_count.emplace_back(out_omp.size());

  KasimtcevOMPMonteCarlo testTaskOmp(taskDataOmp);
  ASSERT_EQ(testTaskOmp.validation(), true);
  testTaskOmp.pre_processing();
  testTaskOmp.run();
  testTaskOmp.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}

TEST(kasimtcev_r_monte_carlo_omp, Tested_x_multiply_yy) {
  double res = 1.5;
  kasimtcev_func f = kasimtcev_fxyy;

  std::vector<double> a1 = {0, 3};
  std::vector<double> a2 = {0, 1};
  std::vector<double> out_seq(1, res);
  std::vector<double> out_omp(1, res);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataSeq->inputs_count.emplace_back(a1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataSeq->inputs_count.emplace_back(a2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(a2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  KasimtcevSequentialMonteCarlo testTaskSeq(taskDataSeq);
  ASSERT_EQ(testTaskSeq.validation(), true);
  testTaskSeq.pre_processing();
  testTaskSeq.run();
  testTaskSeq.post_processing();

  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a1.data()));
  taskDataOmp->inputs_count.emplace_back(a1.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a2.data()));
  taskDataOmp->inputs_count.emplace_back(a2.size());
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataOmp->inputs_count.emplace_back(a2.size());

  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataOmp->outputs_count.emplace_back(out_omp.size());

  KasimtcevOMPMonteCarlo testTaskOmp(taskDataOmp);
  ASSERT_EQ(testTaskOmp.validation(), true);
  testTaskOmp.pre_processing();
  testTaskOmp.run();
  testTaskOmp.post_processing();

  ASSERT_NEAR(out_seq[0], out_omp[0], ESTIMATE);
}
