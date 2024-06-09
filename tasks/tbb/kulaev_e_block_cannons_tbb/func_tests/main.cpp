// Copyright 2024 Kulaev Zhenya
#include <gtest/gtest.h>

#include <vector>

#include "tbb/kulaev_e_block_cannons_tbb/include/ops_tbb.hpp"

TEST(Kulaev_e_block_cannons_tbb, Multiplication_3x3) {
  int n = 3;
  int m = 3;

  std::vector<double> in_A{1, 2, 3, 4, 5, 6, 7, 8, 9};

  // Create data
  std::vector<double> in_B{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> seq_res(n * m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_res.data()));
  taskDataSeq->outputs_count.emplace_back(seq_res.size());

  // Create Task
  TestTBBSequentialKulaevCannon testOmpTaskSequential(taskDataSeq);
  ASSERT_EQ(testOmpTaskSequential.validation(), true);
  testOmpTaskSequential.pre_processing();
  testOmpTaskSequential.run();
  testOmpTaskSequential.post_processing();

  // Create data
  std::vector<double> par_res(n * m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataPar->inputs_count.emplace_back(in_A.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataPar->inputs_count.emplace_back(in_B.size());

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_res.data()));
  taskDataPar->outputs_count.emplace_back(par_res.size());

  // Create Task
  TestTaskTBBParallelKulaevCannon testOmpTaskParallel(taskDataPar);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  for (size_t i = 0; i < seq_res.size(); ++i) {
    ASSERT_EQ(par_res[i], seq_res[i]);
  }
}

TEST(Kulaev_e_block_cannons_tbb, Multiplication_2x2) {
  int n = 2;
  int m = 2;

  std::vector<double> in_A{1, 2, 3, 4};

  // Create data
  std::vector<double> in_B{6, 7, 8, 9};
  std::vector<double> seq_res(n * m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_res.data()));
  taskDataSeq->outputs_count.emplace_back(seq_res.size());

  // Create Task
  TestTBBSequentialKulaevCannon testOmpTaskSequential(taskDataSeq);
  ASSERT_EQ(testOmpTaskSequential.validation(), true);
  testOmpTaskSequential.pre_processing();
  testOmpTaskSequential.run();
  testOmpTaskSequential.post_processing();

  // Create data
  std::vector<double> par_res(n * m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataPar->inputs_count.emplace_back(in_A.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataPar->inputs_count.emplace_back(in_B.size());

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_res.data()));
  taskDataPar->outputs_count.emplace_back(par_res.size());

  // Create Task
  TestTaskTBBParallelKulaevCannon testOmpTaskParallel(taskDataPar);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();
  for (size_t i = 0; i < seq_res.size(); ++i) {
    ASSERT_EQ(par_res[i], seq_res[i]);
  }
}

TEST(Kulaev_e_block_cannons_tbb, Multiplication_5x5) {
  int n = 5;
  int m = 5;

  std::vector<double> in_A = getRandomMatrix(n, m);

  // Create data
  std::vector<double> in_B = getRandomMatrix(n, m);
  std::vector<double> seq_res(n * m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_res.data()));
  taskDataSeq->outputs_count.emplace_back(seq_res.size());

  // Create Task
  TestTBBSequentialKulaevCannon testOmpTaskSequential(taskDataSeq);
  ASSERT_EQ(testOmpTaskSequential.validation(), true);
  testOmpTaskSequential.pre_processing();
  testOmpTaskSequential.run();
  testOmpTaskSequential.post_processing();

  // Create data
  std::vector<double> par_res(n * m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataPar->inputs_count.emplace_back(in_A.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataPar->inputs_count.emplace_back(in_B.size());

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_res.data()));
  taskDataPar->outputs_count.emplace_back(par_res.size());

  // Create Task
  TestTaskTBBParallelKulaevCannon testOmpTaskParallel(taskDataPar);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();
  for (size_t i = 0; i < seq_res.size(); ++i) {
    ASSERT_EQ(par_res[i], seq_res[i]);
  }
}

TEST(Kulaev_e_block_cannons_tbb, Multiplication_0x0) {
  int n = 0;
  int m = 0;

  std::vector<double> in_A = getRandomMatrix(n, m);

  // Create data
  std::vector<double> in_B = getRandomMatrix(n, m);
  std::vector<double> seq_res(n * m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_res.data()));
  taskDataSeq->outputs_count.emplace_back(seq_res.size());

  // Create Task
  TestTBBSequentialKulaevCannon testOmpTaskSequential(taskDataSeq);
  ASSERT_EQ(testOmpTaskSequential.validation(), true);
  testOmpTaskSequential.pre_processing();
  testOmpTaskSequential.run();
  testOmpTaskSequential.post_processing();

  // Create data
  std::vector<double> par_res(n * m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataPar->inputs_count.emplace_back(in_A.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataPar->inputs_count.emplace_back(in_B.size());

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_res.data()));
  taskDataPar->outputs_count.emplace_back(par_res.size());

  // Create Task
  TestTaskTBBParallelKulaevCannon testOmpTaskParallel(taskDataPar);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();
  for (size_t i = 0; i < seq_res.size(); ++i) {
    ASSERT_EQ(par_res[i], seq_res[i]);
  }
}

TEST(Kulaev_e_block_cannons_tbb, Multiplication_100x100) {
  int n = 100;
  int m = 100;

  std::vector<double> in_A = getRandomMatrix(n, m);

  // Create data
  std::vector<double> in_B = getRandomMatrix(n, m);
  std::vector<double> seq_res(n * m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_res.data()));
  taskDataSeq->outputs_count.emplace_back(seq_res.size());

  // Create Task
  TestTBBSequentialKulaevCannon testOmpTaskSequential(taskDataSeq);
  ASSERT_EQ(testOmpTaskSequential.validation(), true);
  testOmpTaskSequential.pre_processing();
  testOmpTaskSequential.run();
  testOmpTaskSequential.post_processing();

  // Create data
  std::vector<double> par_res(n * m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataPar->inputs_count.emplace_back(in_A.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataPar->inputs_count.emplace_back(in_B.size());

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&m));

  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_res.data()));
  taskDataPar->outputs_count.emplace_back(par_res.size());

  // Create Task
  TestTaskTBBParallelKulaevCannon testOmpTaskParallel(taskDataPar);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();
  for (size_t i = 0; i < seq_res.size(); ++i) {
    ASSERT_EQ(par_res[i], seq_res[i]);
  }
}