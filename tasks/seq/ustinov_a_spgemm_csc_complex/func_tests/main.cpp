// Copyright 2024 Ustinov Alexander
#include <gtest/gtest.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>

#include "seq/ustinov_a_spgemm_csc_complex/include/ops_seq.hpp"
#include "seq/ustinov_a_spgemm_csc_complex/include/sparse_matrix.hpp"

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

TEST(ustinov_a_spgemm_csc_complex_seq, test_scalar_matrix) {
  sparse_matrix A(1, 1, 1);
  sparse_matrix B(1, 1, 1);
  sparse_matrix C;
  A.col_ptr = {0, 1};
  A.rows = {0};
  A.values = {std::complex<double>(0.0, 1.0)};
  B.col_ptr = {0, 1};
  B.rows = {0};
  B.values = {std::complex<double>(0.0, -1.0)};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C));

  // Create Task
  SpgemmCSCComplexSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::complex<double> answer(1.0, 0.0);
  ASSERT_NEAR(std::abs(C.values[0] - answer), 0.0, 1e-6);
}

TEST(ustinov_a_spgemm_csc_complex_seq, test_dft2x2) {
  double N = 2.0;
  std::complex<double> exponent{0, -2.0 * PI / N};
  sparse_matrix A(2, 2, 4);
  sparse_matrix B(2, 2, 4);
  sparse_matrix C;
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
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C));

  // Create Task
  SpgemmCSCComplexSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::vector<std::complex<double>> expected_values{{N, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {N, 0.0}};
  for (size_t i = 0; i < C.values.size(); ++i) {
    ASSERT_NEAR(std::abs(C.values[i] - expected_values[i]), 0.0, 1e-6);
  }
}

TEST(ustinov_a_spgemm_csc_complex_seq, test_dft16x16) {
  int n = 16;
  double N = 16.0;
  sparse_matrix A = dft_matrix(n);
  sparse_matrix B = dft_conj_matrix(n);
  sparse_matrix C;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C));

  // Create Task
  SpgemmCSCComplexSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::vector<std::complex<double>> expected_values(n * n);
  for (int i = 0; i < n; ++i) {
    expected_values[i * (n + 1)] = {N, 0.0};
  }
  for (size_t i = 0; i < C.values.size(); ++i) {
    ASSERT_NEAR(std::abs(C.values[i] - expected_values[i]), 0.0, 1e-6);
  }
}

TEST(ustinov_a_spgemm_csc_complex_seq, test_dft64x64) {
  int n = 64;
  double N = 64.0;
  sparse_matrix A = dft_matrix(n);
  sparse_matrix B = dft_conj_matrix(n);
  sparse_matrix C;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C));

  // Create Task
  SpgemmCSCComplexSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::vector<std::complex<double>> expected_values(n * n);
  for (int i = 0; i < n; ++i) {
    expected_values[i * (n + 1)] = {N, 0.0};
  }
  for (size_t i = 0; i < C.values.size(); ++i) {
    ASSERT_NEAR(std::abs(C.values[i] - expected_values[i]), 0.0, 1e-6);
  }
}

TEST(ustinov_a_spgemm_csc_complex_seq, test_shifting_diagonal) {
  int n = 256;
  sparse_matrix A(n, n, n - 1);
  sparse_matrix C;
  for (int i = 0; i < n - 1; ++i) {
    A.col_ptr[i + 1] = i;
    A.rows[i] = i;
    A.values[i] = {1.0, 0.0};
  }

  A.col_ptr[n] = n - 1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C));

  // Create Task
  SpgemmCSCComplexSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(C.col_ptr[n], n - 2);
  EXPECT_EQ(C.col_ptr[0], 0);
  EXPECT_EQ(C.col_ptr[1], 0);
  for (int i = 0; i < n - 2; ++i) {
    EXPECT_EQ(C.col_ptr[i + 2], i);
    EXPECT_EQ(C.rows[i], i);
    EXPECT_NEAR(std::abs(C.values[i] - std::complex<double>(1.0, 0.0)), 0.0, 1e-6);
  }
}

TEST(ustinov_a_spgemm_csc_complex_seq, test_permutation_matrix) {
  int n = 257;
  sparse_matrix A(n, n, n);
  sparse_matrix B(n, n, n);
  sparse_matrix C;
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
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C));

  // Create Task
  SpgemmCSCComplexSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(C.col_ptr[i], i);
    EXPECT_EQ(C.rows[i], i);
    EXPECT_NEAR(std::abs(C.values[i] - std::complex<double>(1.0, 0.0)), 0.0, 1e-6);
  }
  EXPECT_EQ(C.col_ptr[n], n);
}