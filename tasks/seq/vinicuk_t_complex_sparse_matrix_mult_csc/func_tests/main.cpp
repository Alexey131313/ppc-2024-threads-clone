// Copyright 2024 Vinichuk Timofey
#include <gtest/gtest.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>

#include "seq/vinicuk_t_complex_sparse_matrix_mult_csc/include/csc_complex_matrix.hpp"
#include "seq/vinicuk_t_complex_sparse_matrix_mult_csc/include/ops_seq.hpp"

// check main property of complex numbres
TEST(vinicuk_t_complex_sparse_matrix_mult_csc_seq, test_square_of_the_imaginary_part) {
  CSCComplexMatrix mtrx_A(1, 1);
  CSCComplexMatrix mtrx_B(1, 1);
  CSCComplexMatrix mtrx_res(1, 1);

  mtrx_A.col_ptrs = {0, 1};
  mtrx_A.row_indexes = {0};
  mtrx_A.values = {std::complex<double>(0.0, 1.0)};
  mtrx_B.col_ptrs = {0, 1};
  mtrx_B.row_indexes = {0};
  mtrx_B.values = {std::complex<double>(0.0, 1.0)};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_res));

  // Create Task
  MultMatrixCSCComplex testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::complex<double> answer(-1.0, 0.0);
  ASSERT_EQ(mtrx_res.values[0], answer);
}

TEST(vinicuk_t_complex_sparse_matrix_mult_csc_seq, test_multiply_vectors) {
  size_t n = 50;
  CSCComplexMatrix mtrx_A(1, n);
  CSCComplexMatrix mtrx_B(n, 1);
  CSCComplexMatrix mtrx_res(n, n);
  std::complex<double> a(1.0, 1.0);

  mtrx_A.col_ptrs = {0, n};
  // mtrx_B.col_ptrs = {0, 1, 2, 3};
  for (size_t i = 0; i <= n; i++) mtrx_B.col_ptrs.push_back(i);
  for (size_t i = 0; i < n; i++) {
    mtrx_A.row_indexes.push_back((double)i);
    mtrx_A.values.emplace_back(a);
    mtrx_B.row_indexes.push_back(0.0);
    mtrx_B.values.emplace_back(a);
  };

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_res));

  // Create Task
  MultMatrixCSCComplex testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::complex<double> answer(0.0, 2.0);
  for (size_t i = 0; i < n * n; i++) ASSERT_EQ(mtrx_res.values[i], answer);
}

TEST(vinicuk_t_complex_sparse_matrix_mult_csc_seq, test_multiply_diganal_square_matrixs) {
  int n = 50;
  CSCComplexMatrix mtrx_A(n, n);
  CSCComplexMatrix mtrx_B(n, n);
  CSCComplexMatrix mtrx_res(n, n);
  std::complex<double> a(2.0, 1.0);

  for (int i = 0; i <= n; i++) {
    mtrx_A.col_ptrs.push_back(i);
    mtrx_B.col_ptrs.push_back(i);
  }

  for (int i = 0; i < n; i++) {
    mtrx_A.row_indexes.push_back(i);
    mtrx_A.values.emplace_back(a);
    mtrx_B.row_indexes.push_back(i);
    mtrx_B.values.emplace_back(a);
  };

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_res));

  // Create Task
  MultMatrixCSCComplex testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::complex<double> answer1(3.0, 4.0);

  for (size_t i = 0; i < mtrx_res.values.size(); i++) ASSERT_EQ(mtrx_res.values[i], answer1);
}

TEST(vinicuk_t_complex_sparse_matrix_mult_csc_seq, test_multiply_triangle_square_matrix_on_vector) {
  size_t n = 50;
  CSCComplexMatrix mtrx_A(n, n);
  CSCComplexMatrix mtrx_B(1, n);
  CSCComplexMatrix mtrx_res(1, n);
  std::complex<double> a(2.0, 1.0);
  int k = 1;
  int p = 0;
  for (size_t i = 0; i <= n; i++) {
    mtrx_A.col_ptrs.push_back(p);
    p += k;
    k++;
  }

  k = 1;
  p = 0;
  for (size_t i = 0; i < mtrx_A.col_ptrs[n]; i++) {
    mtrx_A.values.emplace_back(a);
    if (p >= k) {
      p = 0;
      k++;
    }
    mtrx_A.row_indexes.push_back(p);
    p++;
  }

  mtrx_B.col_ptrs = {0, n};
  for (size_t i = 0; i < n; i++) {
    mtrx_B.values.emplace_back(a);
    mtrx_B.row_indexes.push_back(i);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_res));

  // Create Task
  MultMatrixCSCComplex testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  k = n;
  for (size_t i = 0; i < mtrx_res.values.size(); i++) {
    ASSERT_EQ(mtrx_res.values[i], std::complex<double>(3.0 * k, 4.0 * k));
    k--;
  }
}

TEST(vinicuk_t_complex_sparse_matrix_mult_csc_seq, test_single_element) {
  CSCComplexMatrix mtrx_A(5, 5);
  CSCComplexMatrix mtrx_B(5, 5);
  CSCComplexMatrix mtrx_res(5, 5);
  std::complex<double> a1(2.0, 1.0);
  std::complex<double> a2(3.0, 4.0);

  mtrx_A.col_ptrs = {0, 0, 1, 1, 1, 1};
  mtrx_B.col_ptrs = {0, 0, 1, 1, 1, 1};

  mtrx_A.values = {a1};
  mtrx_A.row_indexes = {1};
  mtrx_B.values = {a2};
  mtrx_B.row_indexes = {1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&mtrx_res));

  // Create Task
  MultMatrixCSCComplex testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::complex<double> answer1(2.0, 11.0);

  ASSERT_EQ(mtrx_res.values[0], answer1);
}
