
// Copyright 2024 Martynov Aleksandr

#include <gtest/gtest.h>

#include <vector>

#include "seq/martynov_a_strassen_algorithm/include/ops_seq.hpp"

TEST(martynov_a_strassen_alg_seq, multSecShtrassen_4x4) {
  const int n = 4;
  int m = n * n;
  // Create data
  std::vector<double> first_matrix = fillMatrix(n);
  std::vector<double> second_matrix = fillMatrix(n);
  std::vector<double> result(m);
  std::vector<double> res = ijkalgorithm(first_matrix, second_matrix, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(first_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(first_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(second_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(second_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  // Create Task
  Strssn_alg algorithm(taskDataSeq);
  ASSERT_EQ(algorithm.validation(), true);
  algorithm.pre_processing();
  algorithm.run();
  algorithm.post_processing();

  for (size_t i = 0; i < res.size(); i++) {
    ASSERT_DOUBLE_EQ(std::round(res[i]), std::round(result[i]));
  }
}

TEST(martynov_a_strassen_alg_seq, multSecShtrassen_8x8) {
  const int n = 8;
  int m = n * n;
  // Create data
  std::vector<double> first_matrix = fillMatrix(n);
  std::vector<double> second_matrix = fillMatrix(n);
  std::vector<double> result(m);
  std::vector<double> res = ijkalgorithm(first_matrix, second_matrix, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(first_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(first_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(second_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(second_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  // Create Task
  Strssn_alg algorithm(taskDataSeq);
  ASSERT_EQ(algorithm.validation(), true);
  algorithm.pre_processing();
  algorithm.run();
  algorithm.post_processing();

  for (size_t i = 0; i < res.size(); i++) {
    ASSERT_DOUBLE_EQ(std::round(res[i]), std::round(result[i]));
  }
}

TEST(martynov_a_strassen_alg_seq, multSecShtrassen_16x16) {
  const int n = 16;
  int m = n * n;
  // Create data
  std::vector<double> first_matrix = fillMatrix(n);
  std::vector<double> second_matrix = fillMatrix(n);
  std::vector<double> result(m);
  std::vector<double> res = ijkalgorithm(first_matrix, second_matrix, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(first_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(first_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(second_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(second_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  // Create Task
  Strssn_alg algorithm(taskDataSeq);
  ASSERT_EQ(algorithm.validation(), true);
  algorithm.pre_processing();
  algorithm.run();
  algorithm.post_processing();

  for (size_t i = 0; i < res.size(); i++) {
    ASSERT_DOUBLE_EQ(std::round(res[i]), std::round(result[i]));
  }
}

TEST(martynov_a_strassen_alg_seq, multSecShtrassen_32x32) {
  const int n = 32;
  int m = n * n;
  // Create data
  std::vector<double> first_matrix = fillMatrix(n);
  std::vector<double> second_matrix = fillMatrix(n);
  std::vector<double> result(m);
  std::vector<double> res = ijkalgorithm(first_matrix, second_matrix, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(first_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(first_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(second_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(second_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  // Create Task
  Strssn_alg algorithm(taskDataSeq);
  ASSERT_EQ(algorithm.validation(), true);
  algorithm.pre_processing();
  algorithm.run();
  algorithm.post_processing();

  for (size_t i = 0; i < res.size(); i++) {
    ASSERT_DOUBLE_EQ(std::round(res[i]), std::round(result[i]));
  }
}

TEST(martynov_a_strassen_alg_seq, multSecShtrassen_64x64) {
  const int n = 64;
  int m = n * n;
  // Create data
  std::vector<double> first_matrix = fillMatrix(n);
  std::vector<double> second_matrix = fillMatrix(n);
  std::vector<double> result(m);
  std::vector<double> res = ijkalgorithm(first_matrix, second_matrix, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(first_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(first_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(second_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(second_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());
  // Create Task
  Strssn_alg algorithm(taskDataSeq);
  ASSERT_EQ(algorithm.validation(), true);
  algorithm.pre_processing();
  algorithm.run();
  algorithm.post_processing();

  for (size_t i = 0; i < res.size(); i++) {
    ASSERT_DOUBLE_EQ(std::round(res[i]), std::round(result[i]));
  }
}

TEST(martynov_a_strassen_alg_seq, multSecShtrassen_64xsize) {
  const int n = 64;
  int m = n * n;
  // Create data
  std::vector<double> first_matrix = fillMatrix(n);
  std::vector<double> second_matrix = fillMatrix(n);
  std::vector<double> result(m);
  std::vector<double> res = ijkalgorithm(first_matrix, second_matrix, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(first_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(first_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(second_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(second_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  // Create Task
  Strssn_alg algorithm(taskDataSeq);
  ASSERT_EQ(algorithm.validation(), true);
  algorithm.pre_processing();
  algorithm.run();
  algorithm.post_processing();

  for (size_t i = 0; i < res.size(); i++) {
    ASSERT_DOUBLE_EQ(std::round(res[i]), std::round(result[i]));
  }
}
