// Copyright 2024 Ustinov Alexander
#include <gtest/gtest.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>

#include "tbb/ustinov_a_spgemm_csc_complex/include/ops_tbb.hpp"

bool operator==(const sparse_matrix &A, const sparse_matrix &B) {
  if (A.row_num != B.row_num || A.col_num != B.col_num) return false;
  for (int i = 0; i <= A.row_num; ++i) {
    if (A.col_ptr[i] != B.col_ptr[i]) {
      // std::cout << "COL_PTR DOESN'T MATCH: (" << A.col_ptr[i] << " and " << B.col_ptr[i] << "!!!\n";
      return false;
    }
  }
  for (int j = 0; j < A.nonzeros; ++j) {
    if (A.rows[j] != B.rows[j] || std::abs(A.values[j] - B.values[j]) > 1e-15) {
      // if (A.rows[j] != B.rows[j])
      //   std::cout << "ROWS DON'T MATCH: (" << A.rows[j] << " and " << B.rows[j] << "!!!\n";
      // else
      //   std::cout << "VALUES DON'T MATCH: (" << A.values[j] << " and " << B.values[j] << "!!!\n";
      return false;
    }
  }
  return true;
}

const double PI = 3.14159265358979323846;

sparse_matrix dft_matrix(int n) {
  auto N = (double)n;
  std::complex<double> exponent{0.0, -2.0 * PI / N};
  sparse_matrix dft(n, n, n * n);
  for (int i = 1; i <= n; ++i) {
    dft.col_ptr[i] = i * n;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      dft.rows[i * n + j] = j;
      dft.values[i * n + j] = std::exp(exponent * double(i * j));
    }
  }
  return dft;
}

sparse_matrix dft_conj_matrix(int n) {
  auto N = (double)n;
  std::complex<double> exponent{0.0, 2.0 * PI / N};
  sparse_matrix dft_conj(n, n, n * n);
  for (int i = 1; i <= n; ++i) {
    dft_conj.col_ptr[i] = i * n;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      dft_conj.rows[i * n + j] = j;
      dft_conj.values[i * n + j] = std::exp(exponent * double(j * i));
    }
  }
  return dft_conj;
}

TEST(ustinov_a_spgemm_csc_complex_omp, test_scalar_matrix) {
  sparse_matrix A(1, 1, 1);
  sparse_matrix B(1, 1, 1);
  sparse_matrix C_seq;
  sparse_matrix C_par;
  A.col_ptr = {0, 1};
  A.rows = {0};
  A.values = {std::complex<double>(0.0, 1.0)};
  B.col_ptr = {0, 1};
  B.rows = {0};
  B.values = {std::complex<double>(0.0, -1.0)};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C_seq));

  // Create Task
  SpgemmCSCComplexTBBSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C_par));

  // Create Task
  SpgemmCSCComplexTBBPar testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  EXPECT_EQ(C_seq, C_par);
}

TEST(ustinov_a_spgemm_csc_complex_omp, test_dft2x2) {
  double N = 2.0;
  std::complex<double> exponent{0, -2.0 * PI / N};
  sparse_matrix A(2, 2, 4);
  sparse_matrix B(2, 2, 4);
  sparse_matrix C_seq;
  sparse_matrix C_par;
  A.col_ptr = {0, 2, 4};
  A.rows = {0, 1, 0, 1};
  A.values = {std::exp(exponent * 0.0 * 0.0), std::exp(exponent * 0.0 * 1.0), std::exp(exponent * 1.0 * 0.0),
              std::exp(exponent * 1.0 * 1.0)};
  B.col_ptr = {0, 2, 4};
  B.rows = {0, 1, 0, 1};
  B.values = {std::exp(-exponent * 0.0 * 0.0), std::exp(-exponent * 1.0 * 0.0), std::exp(-exponent * 0.0 * 1.0),
              std::exp(-exponent * 1.0 * 1.0)};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C_seq));

  // Create Task
  SpgemmCSCComplexTBBSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C_par));

  // Create Task
  SpgemmCSCComplexTBBPar testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  EXPECT_EQ(C_seq, C_par);
}

TEST(ustinov_a_spgemm_csc_complex_omp, test_dft16x16) {
  int n = 16;
  sparse_matrix A = dft_matrix(n);
  sparse_matrix B = dft_conj_matrix(n);
  sparse_matrix C_seq;
  sparse_matrix C_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C_seq));

  // Create Task
  SpgemmCSCComplexTBBSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C_par));

  // Create Task
  SpgemmCSCComplexTBBPar testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  EXPECT_EQ(C_seq, C_par);
}

TEST(ustinov_a_spgemm_csc_complex_omp, test_dft64x64) {
  int n = 64;
  sparse_matrix A = dft_matrix(n);
  sparse_matrix B = dft_conj_matrix(n);
  sparse_matrix C_seq;
  sparse_matrix C_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C_seq));

  // Create Task
  SpgemmCSCComplexTBBSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C_par));

  // Create Task
  SpgemmCSCComplexTBBPar testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  EXPECT_EQ(C_seq, C_par);
}

TEST(ustinov_a_spgemm_csc_complex_omp, test_shifting_diagonal) {
  int n = 256;
  sparse_matrix A(n, n, n - 1);
  sparse_matrix C_seq;
  sparse_matrix C_par;
  for (int i = 0; i < n - 1; ++i) {
    A.col_ptr[i + 1] = i;
    A.rows[i] = i;
    A.values[i] = {1.0, 0.0};
  }

  A.col_ptr[n] = n - 1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C_seq));

  // Create Task
  SpgemmCSCComplexTBBSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C_par));

  // Create Task
  SpgemmCSCComplexTBBPar testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  EXPECT_EQ(C_seq, C_par);
}

TEST(ustinov_a_spgemm_csc_complex_omp, test_permutation_matrix) {
  int n = 257;
  sparse_matrix A(n, n, n);
  sparse_matrix B(n, n, n);
  sparse_matrix C_seq;
  sparse_matrix C_par;
  int pos = 3;
  for (int i = 0; i < n; ++i) {
    A.col_ptr[i] = B.col_ptr[i] = i;
    A.values[i] = B.values[i] = {1.0, 0.0};
    A.rows[i] = pos;
    pos = (pos * 3) % n;
  }
  A.rows[n - 1] = 0;
  A.col_ptr[n] = B.col_ptr[n] = n;
  for (int i = 0; i < n; ++i) {
    B.rows[A.rows[i]] = i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C_seq));

  // Create Task
  SpgemmCSCComplexTBBSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C_par));

  // Create Task
  SpgemmCSCComplexTBBPar testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  EXPECT_EQ(C_seq, C_par);
}