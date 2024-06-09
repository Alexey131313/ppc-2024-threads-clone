// Copyright 2024 Skotin Alexander
#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include "seq/skotin_a_multiply_matrix_cannon/include/ops_seq.hpp"

std::vector<uint8_t> createMatrixData(size_t size, double value) {
  std::vector<double> matrixData(size * size, value);
  std::vector<uint8_t> byteData(matrixData.size() * sizeof(double));
  memcpy(byteData.data(), matrixData.data(), byteData.size());
  return byteData;
}

void checkMatrixMultiplicationResult(const std::vector<uint8_t>& outputData, size_t matrixSize, double expectedValue) {
  for (size_t i = 0; i < matrixSize; ++i) {
    for (size_t j = 0; j < matrixSize; ++j) {
      double actualValue;
      memcpy(&actualValue, &outputData[(i * matrixSize + j) * sizeof(double)], sizeof(double));
      EXPECT_NEAR(actualValue, expectedValue, 1e-8) << "Mismatch at (" << i << "," << j << ")";
    }
  }
}

void TestMatrixMultiplication(size_t matrixSize, double fillValueA, double fillValueB, double expectedValue) {
  auto matrixAData = createMatrixData(matrixSize, fillValueA);
  auto matrixBData = createMatrixData(matrixSize, fillValueB);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(matrixAData.data());
  taskData->inputs_count.push_back(matrixAData.size());
  taskData->inputs.push_back(matrixBData.data());
  taskData->inputs_count.push_back(matrixBData.size());

  std::vector<uint8_t> outputData(matrixSize * matrixSize * sizeof(double));
  taskData->outputs.push_back(outputData.data());
  taskData->outputs_count.push_back(outputData.size());

  MatrixMultiplicationTask task(taskData);
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  checkMatrixMultiplicationResult(outputData, matrixSize, expectedValue);
}

TEST(Skotin_A_Multiply_Matrix_Cannon_Seq, Test_Multiplication_5x5) { TestMatrixMultiplication(5, 1.0, 2.0, 10.0); }

TEST(Skotin_A_Multiply_Matrix_Cannon_Seq, Test_Multiplication_10x10) { TestMatrixMultiplication(10, 1.0, 2.0, 20.0); }

TEST(Skotin_A_Multiply_Matrix_Cannon_Seq, Test_Multiplication_30x30) { TestMatrixMultiplication(30, 1.0, 2.0, 60.0); }

TEST(Skotin_A_Multiply_Matrix_Cannon_Seq, Test_Multiplication_100x100) {
  TestMatrixMultiplication(100, 1.0, 2.0, 200.0);
}

TEST(Skotin_A_Multiply_Matrix_Cannon_Seq, Test_Multiplication_500x500) {
  TestMatrixMultiplication(500, 1.0, 2.0, 1000.0);
}
