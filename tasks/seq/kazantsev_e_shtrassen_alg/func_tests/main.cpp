// Copyright 2024 Kazantsev Evgeny
#include <gtest/gtest.h>

#include <vector>

#include "seq/kazantsev_e_shtrassen_alg/include/ops_seq.hpp"

TEST(kazantsev_e_matmul_strassen_seq, multSecShtrassen_4x4) {
  const int n = 4;

  // Create data
  std::vector<double> A = getRandomMatrix(n);
  std::vector<double> B = getRandomMatrix(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = multMatrixNoShtrassen(A, B, n);

  // Create Task
  MatMulStrassenSec MatMulStrassenSec(taskDataSeq);
  ASSERT_EQ(MatMulStrassenSec.validation(), true);
  MatMulStrassenSec.pre_processing();
  MatMulStrassenSec.run();
  MatMulStrassenSec.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(std::round(res[i] * std::pow(10, 3)) / std::pow(10, 3),
              std::round(out[i] * std::pow(10, 3)) / std::pow(10, 3));
  }
}

TEST(kazantsev_e_matmul_strassen_seq, multSecShtrassen_8x8) {
  const int n = 8;

  // Create data
  std::vector<double> A = getRandomMatrix(n);
  std::vector<double> B = getRandomMatrix(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = multMatrixNoShtrassen(A, B, n);

  // Create Task
  MatMulStrassenSec MatMulStrassenSec(taskDataSeq);
  ASSERT_EQ(MatMulStrassenSec.validation(), true);
  MatMulStrassenSec.pre_processing();
  MatMulStrassenSec.run();
  MatMulStrassenSec.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(std::round(res[i] * std::pow(10, 3)) / std::pow(10, 3),
              std::round(out[i] * std::pow(10, 3)) / std::pow(10, 3));
  }
}

TEST(kazantsev_e_matmul_strassen_seq, multSecShtrassen_16x16) {
  const int n = 16;

  // Create data
  std::vector<double> A = getRandomMatrix(n);
  std::vector<double> B = getRandomMatrix(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = multMatrixNoShtrassen(A, B, n);

  // Create Task
  MatMulStrassenSec MatMulStrassenSec(taskDataSeq);
  ASSERT_EQ(MatMulStrassenSec.validation(), true);
  MatMulStrassenSec.pre_processing();
  MatMulStrassenSec.run();
  MatMulStrassenSec.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(std::round(res[i] * std::pow(10, 3)) / std::pow(10, 3),
              std::round(out[i] * std::pow(10, 3)) / std::pow(10, 3));
  }
}

TEST(kazantsev_e_matmul_strassen_seq, multSecShtrassen_32x32) {
  const int n = 32;

  // Create data
  std::vector<double> A = getRandomMatrix(n);
  std::vector<double> B = getRandomMatrix(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = multMatrixNoShtrassen(A, B, n);

  // Create Task
  MatMulStrassenSec MatMulStrassenSec(taskDataSeq);
  ASSERT_EQ(MatMulStrassenSec.validation(), true);
  MatMulStrassenSec.pre_processing();
  MatMulStrassenSec.run();
  MatMulStrassenSec.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(std::round(res[i] * std::pow(10, 3)) / std::pow(10, 3),
              std::round(out[i] * std::pow(10, 3)) / std::pow(10, 3));
  }
}

TEST(kazantsev_e_matmul_strassen_seq, multSecShtrassen_64x64) {
  const int n = 64;

  // Create data
  std::vector<double> A = getRandomMatrix(n);
  std::vector<double> B = getRandomMatrix(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = multMatrixNoShtrassen(A, B, n);

  // Create Task
  MatMulStrassenSec MatMulStrassenSec(taskDataSeq);
  ASSERT_EQ(MatMulStrassenSec.validation(), true);
  MatMulStrassenSec.pre_processing();
  MatMulStrassenSec.run();
  MatMulStrassenSec.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(std::round(res[i] * std::pow(10, 3)) / std::pow(10, 3),
              std::round(out[i] * std::pow(10, 3)) / std::pow(10, 3));
  }
}

TEST(kazantsev_e_matmul_strassen_seq, multSecShtrassen_64xsize) {
  const int n = 64;

  // Create data
  std::vector<double> A = getRandomMatrix(n);
  std::vector<double> B = getRandomMatrix(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = multMatrixNoShtrassen(A, B, n);

  // Create Task
  MatMulStrassenSec MatMulStrassenSec(taskDataSeq);
  ASSERT_EQ(MatMulStrassenSec.validation(), true);
  MatMulStrassenSec.pre_processing();
  MatMulStrassenSec.run();
  MatMulStrassenSec.post_processing();

  ASSERT_EQ(res.size(), out.size());
}
