// Copyright 2024 Pozdnyakov Vasya
#include <gtest/gtest.h>
#include <omp.h>

#include <vector>

#include "omp/pozdnyakov_v_rect_integral/include/ops_omp.hpp"

using namespace pozdnyakov_omp;

TEST(pozdnyakov_v_rect_integral_omp, Test_x_sub_y_1) {
  Func f = pozdnyakov_flin;
  int n = 100;

  std::vector<double> in = {0, 3, 0, 3};
  std::vector<double> out_seq(1, 0);
  std::vector<double> out_par(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  PozdnyakovTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
  taskDataPar->outputs_count.emplace_back(out_par.size());

  PozdnyakovTaskOMP testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  ASSERT_EQ(testTaskParallel.pre_processing(), true);
  ASSERT_EQ(testTaskParallel.run(), true);
  ASSERT_EQ(testTaskParallel.post_processing(), true);

  ASSERT_NEAR(out_par[0], out_seq[0], 1e-3);
}

TEST(pozdnyakov_v_rect_integral_omp, Test_x_sub_y_2) {
  Func f = pozdnyakov_flin;
  int n = 100;

  std::vector<double> in = {-5, 2, 4, 10};
  std::vector<double> out_seq(1, 0);
  std::vector<double> out_par(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  PozdnyakovTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
  taskDataPar->outputs_count.emplace_back(out_par.size());

  PozdnyakovTaskOMP testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  ASSERT_EQ(testTaskParallel.pre_processing(), true);
  ASSERT_EQ(testTaskParallel.run(), true);
  ASSERT_EQ(testTaskParallel.post_processing(), true);

  ASSERT_NEAR(out_par[0], out_seq[0], 1e-3);
}

TEST(pozdnyakov_v_rect_integral_omp, Test_x_mul_y) {
  Func f = pozdnyakov_fxy;
  int n = 100;

  std::vector<double> in = {0, 3, 1, 3};
  std::vector<double> out_seq(1, 0);
  std::vector<double> out_par(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  PozdnyakovTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
  taskDataPar->outputs_count.emplace_back(out_par.size());

  PozdnyakovTaskOMP testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  ASSERT_EQ(testTaskParallel.pre_processing(), true);
  ASSERT_EQ(testTaskParallel.run(), true);
  ASSERT_EQ(testTaskParallel.post_processing(), true);

  ASSERT_NEAR(out_par[0], out_seq[0], 1e-3);
}

TEST(pozdnyakov_v_rect_integral_omp, Test_y_sin_x) {
  Func f = pozdnyakov_fysinx;
  int n = 100;

  std::vector<double> in = {-2, 3, 1, 3};
  std::vector<double> out_seq(1, 0);
  std::vector<double> out_par(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  PozdnyakovTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
  taskDataPar->outputs_count.emplace_back(out_par.size());

  PozdnyakovTaskOMP testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  ASSERT_EQ(testTaskParallel.pre_processing(), true);
  ASSERT_EQ(testTaskParallel.run(), true);
  ASSERT_EQ(testTaskParallel.post_processing(), true);

  ASSERT_NEAR(out_par[0], out_seq[0], 1e-3);
}

TEST(pozdnyakov_v_rect_integral_omp, Test_x_exp_y) {
  Func f = pozdnyakov_fxexpy;
  int n = 1000;

  std::vector<double> in = {0, 5, 1, 3};
  std::vector<double> out_seq(1, 0);
  std::vector<double> out_par(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  PozdnyakovTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
  taskDataPar->outputs_count.emplace_back(out_par.size());

  PozdnyakovTaskOMP testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  ASSERT_EQ(testTaskParallel.pre_processing(), true);
  ASSERT_EQ(testTaskParallel.run(), true);
  ASSERT_EQ(testTaskParallel.post_processing(), true);

  ASSERT_NEAR(out_par[0], out_seq[0], 1e-3);
}
