// Copyright 2024 Kirillov Maxim
#include <gtest/gtest.h>

#include <vector>

#include "seq/kirillov_m_strassen_alg/include/ops_seq.hpp"

TEST(kirillov_m_strassen_seq_func_tests, mult4x4) {
  const int n = 4;

  // Create data
  std::vector<double> A = generateRandomMatrixKirillov(n);
  std::vector<double> B = generateRandomMatrixKirillov(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = mulKirillov(A, B, n);

  // Create Task
  StrassenMatrixMultSequential strassenMatrixMultSequential(taskDataSeq);
  ASSERT_EQ(strassenMatrixMultSequential.validation(), true);
  strassenMatrixMultSequential.pre_processing();
  strassenMatrixMultSequential.run();
  strassenMatrixMultSequential.post_processing();

  for (size_t i = 0; i < res.size(); i++) {
    EXPECT_NEAR(res[i], out[i], 10e-6);
  }
}

TEST(kirillov_m_strassen_seq_func_tests, mult8x8) {
  const int n = 8;

  // Create data
  std::vector<double> A = generateRandomMatrixKirillov(n);
  std::vector<double> B = generateRandomMatrixKirillov(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = mulKirillov(A, B, n);

  // Create Task
  StrassenMatrixMultSequential strassenMatrixMultSequential(taskDataSeq);
  ASSERT_EQ(strassenMatrixMultSequential.validation(), true);
  strassenMatrixMultSequential.pre_processing();
  strassenMatrixMultSequential.run();
  strassenMatrixMultSequential.post_processing();

  for (size_t i = 0; i < res.size(); i++) {
    EXPECT_NEAR(res[i], out[i], 10e-6);
  }
}

TEST(kirillov_m_strassen_seq_func_tests, mult16x16) {
  const int n = 16;

  // Create data
  std::vector<double> A = generateRandomMatrixKirillov(n);
  std::vector<double> B = generateRandomMatrixKirillov(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = mulKirillov(A, B, n);

  // Create Task
  StrassenMatrixMultSequential strassenMatrixMultSequential(taskDataSeq);
  ASSERT_EQ(strassenMatrixMultSequential.validation(), true);
  strassenMatrixMultSequential.pre_processing();
  strassenMatrixMultSequential.run();
  strassenMatrixMultSequential.post_processing();

  for (size_t i = 0; i < res.size(); i++) {
    EXPECT_NEAR(res[i], out[i], 10e-6);
  }
}

TEST(kirillov_m_strassen_seq_func_tests, mult32x32) {
  const int n = 32;

  // Create data
  std::vector<double> A = generateRandomMatrixKirillov(n);
  std::vector<double> B = generateRandomMatrixKirillov(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = mulKirillov(A, B, n);

  // Create Task
  StrassenMatrixMultSequential strassenMatrixMultSequential(taskDataSeq);
  ASSERT_EQ(strassenMatrixMultSequential.validation(), true);
  strassenMatrixMultSequential.pre_processing();
  strassenMatrixMultSequential.run();
  strassenMatrixMultSequential.post_processing();

  for (size_t i = 0; i < res.size(); i++) {
    EXPECT_NEAR(res[i], out[i], 10e-6);
  }
}

TEST(kirillov_m_strassen_seq_func_tests, mult64x64) {
  const int n = 64;

  // Create data
  std::vector<double> A = generateRandomMatrixKirillov(n);
  std::vector<double> B = generateRandomMatrixKirillov(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = mulKirillov(A, B, n);

  // Create Task
  StrassenMatrixMultSequential strassenMatrixMultSequential(taskDataSeq);
  ASSERT_EQ(strassenMatrixMultSequential.validation(), true);
  strassenMatrixMultSequential.pre_processing();
  strassenMatrixMultSequential.run();
  strassenMatrixMultSequential.post_processing();

  for (size_t i = 0; i < res.size(); i++) {
    EXPECT_NEAR(res[i], out[i], 10e-6);
  }
}
