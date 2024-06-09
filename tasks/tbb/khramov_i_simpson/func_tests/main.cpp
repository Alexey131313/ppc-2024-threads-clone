// Copyright 2024 Khramov Ivan
#include <gtest/gtest.h>

#include <vector>

#include "tbb/khramov_i_simpson/include/ops_tbb.hpp"

using namespace khramov_tbb;

TEST(khramov_i_simpson_tbb, Test_Linear_Func) {
  int a1 = 0;
  int a2 = 0;
  int b1 = 4;
  int b2 = 4;
  int numSteps = 100;
  double result = 32;
  function func = linear_f;

  // Create data
  std::vector<int> in{a1, a2, b1, b2, numSteps};
  std::vector<double> out{result};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TestSimpsonTbbsequential testTaskSequential(taskDataSeq, func);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);

  std::vector<double> par_res{result};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_res.data()));
  taskDataPar->outputs_count.emplace_back(par_res.size());

  // Create Task
  TestSimpsonTbbParallel testTaskParallel(taskDataPar, func);
  ASSERT_EQ(testTaskParallel.validation(), true);
  ASSERT_EQ(testTaskParallel.pre_processing(), true);
  ASSERT_EQ(testTaskParallel.run(), true);
  ASSERT_EQ(testTaskParallel.post_processing(), true);

  ASSERT_NEAR(par_res[0], out[0], 1e-3);
}

TEST(khramov_i_simpson_tbb, Test_Square_Func) {
  int a1 = 0;
  int a2 = 0;
  int b1 = 3;
  int b2 = 3;
  int numSteps = 100;
  double result = 81;
  function func = square_f;

  // Create data
  std::vector<int> in{a1, a2, b1, b2, numSteps};
  std::vector<double> out{result};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TestSimpsonTbbsequential testTaskSequential(taskDataSeq, func);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);

  std::vector<double> par_res{result};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_res.data()));
  taskDataPar->outputs_count.emplace_back(par_res.size());

  // Create Task
  TestSimpsonTbbParallel testTaskParallel(taskDataPar, func);
  ASSERT_EQ(testTaskParallel.validation(), true);
  ASSERT_EQ(testTaskParallel.pre_processing(), true);
  ASSERT_EQ(testTaskParallel.run(), true);
  ASSERT_EQ(testTaskParallel.post_processing(), true);

  ASSERT_NEAR(par_res[0], out[0], 1e-3);
}

TEST(khramov_i_simpson_tbb, Test_Sin_Func) {
  int a1 = 0;
  int a2 = 0;
  int b1 = 3;
  int b2 = 3;
  int numSteps = 100;
  double result = 0.561;
  function func = sin_f;

  // Create data
  std::vector<int> in{a1, a2, b1, b2, numSteps};
  std::vector<double> out{result};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TestSimpsonTbbsequential testTaskSequential(taskDataSeq, func);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);

  std::vector<double> par_res{result};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_res.data()));
  taskDataPar->outputs_count.emplace_back(par_res.size());

  // Create Task
  TestSimpsonTbbParallel testTaskParallel(taskDataPar, func);
  ASSERT_EQ(testTaskParallel.validation(), true);
  ASSERT_EQ(testTaskParallel.pre_processing(), true);
  ASSERT_EQ(testTaskParallel.run(), true);
  ASSERT_EQ(testTaskParallel.post_processing(), true);

  ASSERT_NEAR(par_res[0], out[0], 1e-3);
}

TEST(khramov_i_simpson_tbb, Test_Mult_Func) {
  int a1 = 0;
  int a2 = 0;
  int b1 = 3;
  int b2 = 3;
  int numSteps = 100;
  double result = 20.25;
  function func = mult_f;

  // Create data
  std::vector<int> in{a1, a2, b1, b2, numSteps};
  std::vector<double> out{result};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TestSimpsonTbbsequential testTaskSequential(taskDataSeq, func);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);

  std::vector<double> par_res{result};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_res.data()));
  taskDataPar->outputs_count.emplace_back(par_res.size());

  // Create Task
  TestSimpsonTbbParallel testTaskParallel(taskDataPar, func);
  ASSERT_EQ(testTaskParallel.validation(), true);
  ASSERT_EQ(testTaskParallel.pre_processing(), true);
  ASSERT_EQ(testTaskParallel.run(), true);
  ASSERT_EQ(testTaskParallel.post_processing(), true);

  ASSERT_NEAR(par_res[0], out[0], 1e-3);
}

TEST(khramov_i_simpson_tbb, Test_Exp_Func) {
  int a1 = 0;
  int a2 = 0;
  int b1 = 3;
  int b2 = 3;
  int numSteps = 100;
  double result = 1035.103;
  function func = exp_f;

  // Create data
  std::vector<int> in{a1, a2, b1, b2, numSteps};
  std::vector<double> out{result};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  TestSimpsonTbbsequential testTaskSequential(taskDataSeq, func);
  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_EQ(testTaskSequential.pre_processing(), true);
  ASSERT_EQ(testTaskSequential.run(), true);
  ASSERT_EQ(testTaskSequential.post_processing(), true);

  std::vector<double> par_res{result};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_res.data()));
  taskDataPar->outputs_count.emplace_back(par_res.size());

  // Create Task
  TestSimpsonTbbParallel testTaskParallel(taskDataPar, func);
  ASSERT_EQ(testTaskParallel.validation(), true);
  ASSERT_EQ(testTaskParallel.pre_processing(), true);
  ASSERT_EQ(testTaskParallel.run(), true);
  ASSERT_EQ(testTaskParallel.post_processing(), true);

  ASSERT_NEAR(par_res[0], out[0], 1e-3);
}
