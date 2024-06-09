// Copyright 2024 Bodrov Daniil
#include <gtest/gtest.h>

#include <complex>
#include <vector>

#include "seq/bodrov_d_crs_matr/include/bodrov_d_crs_matr_seq.hpp"

TEST(bodrov_d_crs_matr_seq, test_invalid_matrices) {
  SparseMatrix A;
  A.n_rows = 2;
  A.n_cols = 3;
  A.pointer = {0, 2, 3};
  A.col_indexes = {0, 2, 1};
  A.non_zero_values = {1, 2, 3};

  SparseMatrix B;
  B.n_rows = 3;
  B.n_cols = 2;
  B.pointer = {0, 2, 3};
  B.col_indexes = {0, 2, 1};
  B.non_zero_values = {1, 2, 3};

  SparseMatrix Result;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Result));

  // Create Task
  SparseMatrixSolver taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), false);
}

TEST(bodrov_d_crs_matr_seq, test_identity_matrices) {
  SparseMatrix A;
  A.n_rows = 3;
  A.n_cols = 3;
  A.pointer = {0, 1, 2, 3};
  A.col_indexes = {0, 1, 2};
  A.non_zero_values = {1, 1, 1};

  SparseMatrix B;
  B.n_rows = 3;
  B.n_cols = 3;
  B.pointer = {0, 1, 2, 3};
  B.col_indexes = {0, 1, 2};
  B.non_zero_values = {1, 1, 1};

  SparseMatrix Result;

  SparseMatrix Expected;
  Expected.n_rows = 3;
  Expected.n_cols = 3;
  Expected.pointer = {0, 1, 2, 3};
  Expected.col_indexes = {0, 1, 2};
  Expected.non_zero_values = {1, 1, 1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Result));

  // Create Task
  SparseMatrixSolver taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  ASSERT_EQ(taskSequential.pre_processing(), true);
  ASSERT_EQ(taskSequential.run(), true);
  ASSERT_EQ(taskSequential.post_processing(), true);

  ASSERT_EQ(Result.n_rows, Expected.n_rows);
  ASSERT_EQ(Result.n_cols, Expected.n_cols);
  ASSERT_EQ(Result.pointer, Expected.pointer);
  ASSERT_EQ(Result.col_indexes, Expected.col_indexes);
  ASSERT_EQ(Result.non_zero_values.size(), Expected.non_zero_values.size());
}

TEST(bodrov_d_crs_matr_seq, test_sparse_matrices) {
  SparseMatrix A;
  A.n_rows = 3;
  A.n_cols = 3;
  A.pointer = {0, 1, 3, 4};
  A.col_indexes = {0, 1, 0, 2};
  A.non_zero_values = {1, 2, 3, 4};

  SparseMatrix B;
  B.n_rows = 3;
  B.n_cols = 3;
  B.pointer = {0, 2, 3, 4};
  B.col_indexes = {0, 2, 1, 2};
  B.non_zero_values = {1, 2, 3, 4};

  SparseMatrix Result;

  SparseMatrix Expected;
  Expected.n_rows = 3;
  Expected.n_cols = 3;
  Expected.pointer = {0, 2, 3, 4};
  Expected.col_indexes = {0, 2, 1, 2};
  Expected.non_zero_values = {1, 2, 6, 16};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Result));

  // Create Task
  SparseMatrixSolver taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  ASSERT_EQ(taskSequential.pre_processing(), true);
  ASSERT_EQ(taskSequential.run(), true);
  ASSERT_EQ(taskSequential.post_processing(), true);

  ASSERT_EQ(Result.n_rows, Expected.n_rows);
  ASSERT_EQ(Result.n_cols, Expected.n_cols);
  ASSERT_EQ(Result.pointer, Expected.pointer);
  ASSERT_EQ(Result.col_indexes, Expected.col_indexes);
  ASSERT_EQ(Result.non_zero_values.size(), Expected.non_zero_values.size());
  for (size_t i = 0; i < Expected.non_zero_values.size(); i++) {
    std::complex<double> t = Expected.non_zero_values[i] - Result.non_zero_values[i];
    ASSERT_NEAR(0.0f, t.imag(), 1e-3);
    ASSERT_NEAR(0.0f, t.real(), 1e-3);
  }
}

TEST(bodrov_d_crs_matr_seq, test_inverse_matrix) {
  SparseMatrix A;
  A.n_rows = 4;
  A.n_cols = 4;
  A.pointer = {0, 1, 3, 5, 8};
  A.col_indexes = {0, 1, 3, 0, 2, 0, 1, 3};
  A.non_zero_values = {1, {0, 1}, {0, 1}, 1, {1, 1}, 2, 3, {2, -1}};

  SparseMatrix B;
  B.n_rows = 4;
  B.n_cols = 4;
  B.pointer = {0, 1, 4, 6, 9};
  B.col_indexes = {0, 0, 1, 3, 0, 2, 0, 1, 3};
  B.non_zero_values = {1,           {-1, 1}, {1.5, 0.5},   {0.5, -0.5}, {-0.5, 0.5},
                       {0.5, -0.5}, {1, -1}, {-1.5, -1.5}, {-0.5, 0.5}};

  SparseMatrix Result;
  SparseMatrix Expected;
  Expected.n_rows = 4;
  Expected.n_cols = 4;
  Expected.pointer = {0, 1, 2, 3, 4};
  Expected.col_indexes = {0, 1, 2, 3};
  Expected.non_zero_values = {1, 1, 1, 1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Result));

  // Create Task
  SparseMatrixSolver taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  ASSERT_EQ(taskSequential.pre_processing(), true);
  ASSERT_EQ(taskSequential.run(), true);
  ASSERT_EQ(taskSequential.post_processing(), true);

  ASSERT_EQ(Result.n_rows, Expected.n_rows);
  ASSERT_EQ(Result.n_cols, Expected.n_cols);
  ASSERT_EQ(Result.pointer, Expected.pointer);
  ASSERT_EQ(Result.col_indexes, Expected.col_indexes);
  ASSERT_EQ(Result.non_zero_values.size(), Expected.non_zero_values.size());
  for (size_t i = 0; i < Expected.non_zero_values.size(); i++) {
    std::complex<double> t = Expected.non_zero_values[i] - Result.non_zero_values[i];
    ASSERT_NEAR(0.0f, t.imag(), 1e-3);
    ASSERT_NEAR(0.0f, t.real(), 1e-3);
  }
}

TEST(bodrov_d_crs_matr_seq, test_complex_matrix) {
  SparseMatrix A;
  A.n_rows = 5;
  A.n_cols = 4;
  A.pointer = {0, 2, 3, 4, 5, 6};
  A.col_indexes = {0, 2, 1, 0, 2, 3};
  A.non_zero_values = {{0, 1}, {2, 3}, {1, 1}, {4, 1}, {1, 2}, {2, 2}};

  SparseMatrix B;
  B.n_rows = 4;
  B.n_cols = 5;
  B.pointer = {0, 2, 3, 5, 7};
  B.col_indexes = {0, 3, 2, 1, 3, 2, 4};
  B.non_zero_values = {{1, 1}, {4, 1}, {1, 2}, {3, 0}, {7, 7}, {2, 1}, {3, 2}};

  SparseMatrix Result;

  SparseMatrix Expected;
  Expected.n_rows = 5;
  Expected.n_cols = 5;
  Expected.pointer = {0, 3, 4, 6, 8, 10};
  Expected.col_indexes = {0, 1, 3, 2, 0, 3, 1, 3, 2, 4};
  Expected.non_zero_values = {{-1, 1}, {6, 9}, {-8, 39}, {-1, 3}, {3, 5}, {15, 8}, {3, 6}, {-7, 21}, {2, 6}, {2, 10}};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Result));

  // Create Task
  SparseMatrixSolver taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  ASSERT_EQ(taskSequential.pre_processing(), true);
  ASSERT_EQ(taskSequential.run(), true);
  ASSERT_EQ(taskSequential.post_processing(), true);

  ASSERT_EQ(Result.n_rows, Expected.n_rows);
  ASSERT_EQ(Result.n_cols, Expected.n_cols);
  ASSERT_EQ(Result.pointer, Expected.pointer);
  ASSERT_EQ(Result.col_indexes, Expected.col_indexes);
  ASSERT_EQ(Result.non_zero_values.size(), Expected.non_zero_values.size());
  for (size_t i = 0; i < Expected.non_zero_values.size(); i++) {
    std::complex<double> t = Expected.non_zero_values[i] - Result.non_zero_values[i];
    ASSERT_NEAR(0.0f, t.imag(), 1e-3);
    ASSERT_NEAR(0.0f, t.real(), 1e-3);
  }
}

TEST(bodrov_d_crs_matr_seq, test_complex_matrix_diagonal) {
  SparseMatrix A;
  A.n_rows = 3;
  A.n_cols = 3;
  A.pointer = {0, 1, 2, 3};
  A.col_indexes = {0, 1, 2};
  A.non_zero_values = {{2, 3}, {1, 1}, {4, 1}};

  SparseMatrix B;
  B.n_rows = 3;
  B.n_cols = 3;
  B.pointer = {0, 1, 2, 3};
  B.col_indexes = {2, 1, 0};
  B.non_zero_values = {{3, 0}, {7, 7}, {2, 1}};

  SparseMatrix Result;

  SparseMatrix Expected;
  Expected.n_rows = 3;
  Expected.n_cols = 3;
  Expected.pointer = {0, 1, 2, 3};
  Expected.col_indexes = {2, 1, 0};
  Expected.non_zero_values = {{6, 9}, {0, 14}, {7, 6}};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&Result));

  // Create Task
  SparseMatrixSolver taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  ASSERT_EQ(taskSequential.pre_processing(), true);
  ASSERT_EQ(taskSequential.run(), true);
  ASSERT_EQ(taskSequential.post_processing(), true);

  ASSERT_EQ(Result.n_rows, Expected.n_rows);
  ASSERT_EQ(Result.n_cols, Expected.n_cols);
  ASSERT_EQ(Result.pointer, Expected.pointer);
  ASSERT_EQ(Result.col_indexes, Expected.col_indexes);
  ASSERT_EQ(Result.non_zero_values.size(), Expected.non_zero_values.size());
  for (size_t i = 0; i < Expected.non_zero_values.size(); i++) {
    std::complex<double> t = Expected.non_zero_values[i] - Result.non_zero_values[i];
    ASSERT_NEAR(0.0f, t.imag(), 1e-3);
    ASSERT_NEAR(0.0f, t.real(), 1e-3);
  }
}