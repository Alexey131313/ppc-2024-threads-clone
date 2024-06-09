// Copyright 2023 Belan Vadim
#include <gtest/gtest.h>

#include <vector>

#include "seq/belan_vadim_mat_fox_alg/include/ops_seq.hpp"

TEST(MatrixMulFox, Test_Matrix_Mul_1x1) {
  const int n = 1;
  const int block_size = 1;

  std::vector<int> input(2, 0);
  input[0] = n;
  input[1] = block_size;

  std::vector<double> matrix_a(n * n, 2.0);
  std::vector<double> matrix_b(n * n, 3.0);
  std::vector<double> matrix_c(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  taskData->inputs_count.emplace_back(matrix_a.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  taskData->inputs_count.emplace_back(matrix_b.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  taskData->outputs_count.emplace_back(matrix_c.size());

  MatrixMulFox matrixMulFox(taskData);
  ASSERT_EQ(matrixMulFox.validation(), true);
  matrixMulFox.pre_processing();
  matrixMulFox.run();
  matrixMulFox.post_processing();

  std::vector<double> expected_result(n * n, 6.0);
  for (int i = 0; i < n * n; i++) {
    ASSERT_DOUBLE_EQ(matrix_c[i], expected_result[i]);
  }
}

TEST(MatrixMulFox, Test_Matrix_Mul_2x2) {
  const int n = 2;
  const int block_size = 2;

  std::vector<int> input(2, 0);
  input[0] = n;
  input[1] = block_size;

  std::vector<double> matrix_a(n * n, 1.0);
  std::vector<double> matrix_b(n * n, 2.0);
  std::vector<double> matrix_c(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  taskData->inputs_count.emplace_back(matrix_a.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  taskData->inputs_count.emplace_back(matrix_b.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  taskData->outputs_count.emplace_back(matrix_c.size());

  MatrixMulFox matrixMulFox(taskData);
  ASSERT_EQ(matrixMulFox.validation(), true);
  matrixMulFox.pre_processing();
  matrixMulFox.run();
  matrixMulFox.post_processing();

  std::vector<double> expected_result(n * n, 4.0);
  ASSERT_EQ(matrix_c, expected_result);
}

TEST(MatrixMulFox, Test_Matrix_Mul_3x3) {
  const int n = 3;
  const int block_size = 3;

  std::vector<int> input(2, 0);
  input[0] = n;
  input[1] = block_size;

  std::vector<double> matrix_a(n * n, 1.0);
  std::vector<double> matrix_b(n * n, 2.0);
  std::vector<double> matrix_c(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  taskData->inputs_count.emplace_back(matrix_a.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  taskData->inputs_count.emplace_back(matrix_b.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  taskData->outputs_count.emplace_back(matrix_c.size());

  MatrixMulFox matrixMulFox(taskData);
  ASSERT_EQ(matrixMulFox.validation(), true);
  matrixMulFox.pre_processing();
  matrixMulFox.run();
  matrixMulFox.post_processing();

  std::vector<double> expected_result(n * n, 6.0);
  for (int i = 0; i < n * n; i++) {
    ASSERT_DOUBLE_EQ(matrix_c[i], expected_result[i]);
  }
}

TEST(MatrixMulFox, Test_Matrix_Mul_4x4) {
  const int n = 4;
  const int block_size = 4;

  std::vector<int> input(2, 0);
  input[0] = n;
  input[1] = block_size;

  std::vector<double> matrix_a(n * n, 1.0);
  std::vector<double> matrix_b(n * n, 2.0);
  std::vector<double> matrix_c(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  taskData->inputs_count.emplace_back(matrix_a.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  taskData->inputs_count.emplace_back(matrix_b.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  taskData->outputs_count.emplace_back(matrix_c.size());

  MatrixMulFox matrixMulFox(taskData);
  ASSERT_EQ(matrixMulFox.validation(), true);
  matrixMulFox.pre_processing();
  matrixMulFox.run();
  matrixMulFox.post_processing();

  std::vector<double> expected_result(n * n, 8.0);
  for (int i = 0; i < n * n; i++) {
    ASSERT_DOUBLE_EQ(matrix_c[i], expected_result[i]);
  }
}

TEST(MatrixMulFox, Test_Matrix_Mul_5x5) {
  const int n = 5;
  const int block_size = 5;

  std::vector<int> input(2, 0);
  input[0] = n;
  input[1] = block_size;

  std::vector<double> matrix_a(n * n, 1.0);
  std::vector<double> matrix_b(n * n, 2.0);
  std::vector<double> matrix_c(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  taskData->inputs_count.emplace_back(matrix_a.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  taskData->inputs_count.emplace_back(matrix_b.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  taskData->outputs_count.emplace_back(matrix_c.size());

  MatrixMulFox matrixMulFox(taskData);
  ASSERT_EQ(matrixMulFox.validation(), true);
  matrixMulFox.pre_processing();
  matrixMulFox.run();
  matrixMulFox.post_processing();

  std::vector<double> expected_result(n * n, 10.0);
  for (int i = 0; i < n * n; i++) {
    ASSERT_DOUBLE_EQ(matrix_c[i], expected_result[i]);
  }
}
