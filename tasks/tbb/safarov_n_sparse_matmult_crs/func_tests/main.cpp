// Copyright 2024 Safarov Nurlan
#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "tbb/safarov_n_sparse_matmult_crs/include/sparse_matmult_crs_tbb.hpp"

TEST(Safarov_N_SparseMatMultCRS_TBB, TestOne) {
  // Create data
  SparseMatrixCRS x(6, 6, {1, 2, 3, 4, 2, 5, 7}, {0, 4, 2, 3, 3, 5, 1}, {0, 2, 4, 4, 6, 6, 7});
  SparseMatrixCRS y(6, 6, {3, 1, 1, 3, 1, 3, 2, 4}, {0, 5, 2, 3, 0, 2, 1, 4}, {0, 2, 3, 3, 4, 6, 8});
  SparseMatrixCRS correctAnswer(6, 6, {5, 6, 1, 12, 10, 6, 20, 7}, {0, 2, 5, 3, 1, 3, 4, 2}, {0, 3, 4, 4, 7, 7, 8});
  SparseMatrixCRS z;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&x));
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&y));
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&z));

  // Create Task
  SparseMatrixMultiplicationCRS_TBB taskTbb(taskDataTbb);
  ASSERT_EQ(taskTbb.validation(), true);
  ASSERT_EQ(taskTbb.pre_processing(), true);
  ASSERT_EQ(taskTbb.run(), true);
  ASSERT_EQ(taskTbb.post_processing(), true);

  ASSERT_EQ(z.numberOfRows, correctAnswer.numberOfRows);
  ASSERT_EQ(z.numberOfColumns, correctAnswer.numberOfColumns);
  ASSERT_EQ(z.values.size(), correctAnswer.values.size());
  ASSERT_EQ(z, correctAnswer);
  // ASSERT_EQ(z.values, correctAnswer.values);
  for (size_t i = 0; i < correctAnswer.values.size(); ++i) {
    double t = correctAnswer.values[i] - z.values[i];
    ASSERT_NEAR(0.0f, t, 1e-6);
  }
}

TEST(Safarov_N_SparseMatMultCRS_TBB, TestTwo) {
  // Create data
  std::vector<std::vector<double>> temporaryX = createRandomMatrix(3, 3, 0.4);
  std::vector<std::vector<double>> temporaryY = createRandomMatrix(3, 3, 0.4);
  SparseMatrixCRS x(temporaryX);
  SparseMatrixCRS y(temporaryY);
  std::vector<std::vector<double>> temporaryXMultiplyBytemporaryY = multiplyMatrices(temporaryX, temporaryY);
  SparseMatrixCRS temporaryMatrix(temporaryXMultiplyBytemporaryY);
  SparseMatrixCRS z;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&x));
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&y));
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&z));

  // Create Task
  SparseMatrixMultiplicationCRS_TBB taskTbb(taskDataTbb);
  ASSERT_EQ(taskTbb.validation(), true);
  ASSERT_EQ(taskTbb.pre_processing(), true);
  ASSERT_EQ(taskTbb.run(), true);
  ASSERT_EQ(taskTbb.post_processing(), true);

  ASSERT_EQ(z.numberOfRows, temporaryMatrix.numberOfRows);
  ASSERT_EQ(z.numberOfColumns, temporaryMatrix.numberOfColumns);
  ASSERT_EQ(z.values.size(), temporaryMatrix.values.size());
  ASSERT_EQ(z, temporaryMatrix);
  // ASSERT_EQ(z.values, temporaryMatrix.values);
  for (size_t i = 0; i < temporaryMatrix.values.size(); ++i) {
    double t = temporaryMatrix.values[i] - z.values[i];
    ASSERT_NEAR(0.0f, t, 1e-6);
  }
}

TEST(Safarov_N_SparseMatMultCRS_TBB, TestThree) {
  // Create data
  SparseMatrixCRS x;
  x.numberOfRows = 5;
  x.numberOfColumns = 5;
  x.pointers = {0, 2, 3, 3, 4, 5};
  x.columnIndexes = {0, 2, 1, 2, 1};
  x.values = {2.5, -1.7, 3.4, 0.8, 0.6};

  SparseMatrixCRS identityMatrix;
  identityMatrix.numberOfRows = 5;
  identityMatrix.numberOfColumns = 5;
  identityMatrix.pointers = {0, 1, 2, 3, 4, 5};
  identityMatrix.columnIndexes = {0, 1, 2, 3, 4};
  identityMatrix.values = {1.0, 1.0, 1.0, 1.0, 1.0};

  SparseMatrixCRS oldX;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&x));
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&identityMatrix));
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&oldX));

  // Create Task
  SparseMatrixMultiplicationCRS_TBB taskTbb(taskDataTbb);
  ASSERT_EQ(taskTbb.validation(), true);
  ASSERT_EQ(taskTbb.pre_processing(), true);
  ASSERT_EQ(taskTbb.run(), true);
  ASSERT_EQ(taskTbb.post_processing(), true);

  SparseMatrixCRS correctAnswer;
  correctAnswer.numberOfRows = 5;
  correctAnswer.numberOfColumns = 5;
  correctAnswer.pointers = {0, 2, 3, 3, 4, 5};
  correctAnswer.columnIndexes = {0, 2, 1, 2, 1};
  correctAnswer.values = {2.5, -1.7, 3.4, 0.8, 0.6};

  ASSERT_EQ(oldX.numberOfRows, correctAnswer.numberOfRows);
  ASSERT_EQ(oldX.numberOfColumns, correctAnswer.numberOfColumns);
  ASSERT_EQ(oldX.values.size(), correctAnswer.values.size());
  ASSERT_EQ(oldX, correctAnswer);
  // ASSERT_EQ(oldX.values, correctAnswer.values);
  for (size_t i = 0; i < correctAnswer.values.size(); ++i) {
    double t = correctAnswer.values[i] - oldX.values[i];
    ASSERT_NEAR(0.0f, t, 1e-6);
  }
}

TEST(Safarov_N_SparseMatMultCRS_TBB, TestFour) {
  // Create data
  SparseMatrixCRS x;
  x.numberOfRows = 4;
  x.numberOfColumns = 5;
  x.pointers = {0, 2, 4, 5, 7};
  x.columnIndexes = {0, 3, 2, 4, 1, 0, 3};
  x.values = {3.9, 1.2, 10.45, 5.1, -2.3, 8.6, -0.7};

  SparseMatrixCRS y;
  y.numberOfRows = 5;
  y.numberOfColumns = 4;
  y.pointers = {0, 1, 2, 3, 5, 6};
  y.columnIndexes = {2, 1, 2, 0, 3, 1};
  y.values = {-1.1, 0.5, 3.1, 0.8, -2.7, 1.4};
  SparseMatrixCRS z;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&x));
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&y));
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&z));

  // Create Task
  SparseMatrixMultiplicationCRS_TBB taskTbb(taskDataTbb);
  ASSERT_EQ(taskTbb.validation(), true);
  ASSERT_EQ(taskTbb.pre_processing(), true);
  ASSERT_EQ(taskTbb.run(), true);
  ASSERT_EQ(taskTbb.post_processing(), true);

  SparseMatrixCRS correctAnswer;
  correctAnswer.numberOfRows = 4;
  correctAnswer.numberOfColumns = 4;
  correctAnswer.pointers = {0, 3, 5, 6, 9};
  correctAnswer.columnIndexes = {0, 2, 3, 1, 2, 1, 0, 2, 3};
  correctAnswer.values = {0.96, -4.29, -3.24, 7.14, 32.395, -1.15, -0.56, -9.46, 1.89};

  ASSERT_EQ(z.numberOfRows, correctAnswer.numberOfRows);
  ASSERT_EQ(z.numberOfColumns, correctAnswer.numberOfColumns);
  ASSERT_EQ(z.pointers, correctAnswer.pointers);
  ASSERT_EQ(z.columnIndexes, correctAnswer.columnIndexes);
  ASSERT_EQ(z.values.size(), correctAnswer.values.size());
  // ASSERT_EQ(z.values, correctAnswer.values);
  for (size_t i = 0; i < correctAnswer.values.size(); ++i) {
    double t = correctAnswer.values[i] - z.values[i];
    ASSERT_NEAR(0.0f, t, 1e-6);
  }
}

TEST(Safarov_N_SparseMatMultCRS_TBB, TestFive) {
  // Create data
  SparseMatrixCRS x;
  x.numberOfRows = 3;
  x.numberOfColumns = 4;
  x.pointers = {0, 1, 2, 3};
  x.columnIndexes = {0, 2, 3};
  x.values = {7.2, 10.5, 13.1};

  SparseMatrixCRS y;
  y.numberOfRows = 4;
  y.numberOfColumns = 3;
  y.pointers = {0, 1, 1, 2, 3};
  y.columnIndexes = {2, 0, 1};
  y.values = {8.3, 5.7, 12.4};
  SparseMatrixCRS z;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&x));
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&y));
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(&z));

  // Create Task
  SparseMatrixMultiplicationCRS_TBB taskTbb(taskDataTbb);
  ASSERT_EQ(taskTbb.validation(), true);
  ASSERT_EQ(taskTbb.pre_processing(), true);
  ASSERT_EQ(taskTbb.run(), true);
  ASSERT_EQ(taskTbb.post_processing(), true);

  SparseMatrixCRS correctAnswer;
  correctAnswer.numberOfRows = 3;
  correctAnswer.numberOfColumns = 3;
  correctAnswer.pointers = {0, 1, 2, 3};
  correctAnswer.columnIndexes = {2, 0, 1};
  correctAnswer.values = {59.76, 59.85, 162.44};

  ASSERT_EQ(z.numberOfRows, correctAnswer.numberOfRows);
  ASSERT_EQ(z.numberOfColumns, correctAnswer.numberOfColumns);
  ASSERT_EQ(z.pointers, correctAnswer.pointers);
  ASSERT_EQ(z.columnIndexes, correctAnswer.columnIndexes);
  ASSERT_EQ(z.values.size(), correctAnswer.values.size());
  // ASSERT_EQ(z.values, correctAnswer.values);
  for (size_t i = 0; i < correctAnswer.values.size(); ++i) {
    double t = correctAnswer.values[i] - z.values[i];
    ASSERT_NEAR(0.0f, t, 1e-6);
  }
}