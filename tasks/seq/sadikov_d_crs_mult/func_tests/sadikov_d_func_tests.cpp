// Copyright 2024 Sadikov Damir
#include <gtest/gtest.h>

#include <complex>
#include <numeric>
#include <vector>

#include "seq/sadikov_d_crs_mult/include/sadikov_d_seq.hpp"

TEST(sadikov_d_crs_mult_seq, test_small_real_square_matrix) {
  matrix_CRS A;
  A.n = 5;
  A.m = 5;
  A.row_id = {0, 1, 4, 4, 6, 8};
  A.col = {0, 1, 3, 4, 0, 4, 1, 3};
  A.value = {1, 2, 4, 3, 1, -3, 7, 6};
  matrix_CRS B;
  B.n = 5;
  B.m = 5;
  B.row_id = {0, 1, 3, 4, 6, 7};
  B.col = {3, 1, 2, 3, 0, 1, 2};
  B.value = {1, 1, 4, -2, 4, 3, 1};
  matrix_CRS C;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C));

  // Create Task
  CRSComplexMult_Sequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  ASSERT_EQ(taskSequential.pre_processing(), true);
  ASSERT_EQ(taskSequential.run(), true);
  ASSERT_EQ(taskSequential.post_processing(), true);

  // Check Result
  matrix_CRS C_expected;
  C_expected.n = 5;
  C_expected.m = 5;
  C_expected.row_id = {0, 1, 4, 4, 6, 9};
  C_expected.col = {3, 0, 1, 2, 2, 3, 0, 1, 2};
  C_expected.value = {1, 16, 14, 11, -3, 1, 24, 25, 28};

  ASSERT_EQ(C.n, C_expected.n);
  ASSERT_EQ(C.m, C_expected.m);
  ASSERT_EQ(C.row_id, C_expected.row_id);
  ASSERT_EQ(C.col, C_expected.col);
  ASSERT_EQ(C.value.size(), C_expected.value.size());
  for (size_t i = 0; i < C_expected.value.size(); i++) {
    std::complex<double> t = C_expected.value[i] - C.value[i];
    ASSERT_NEAR(0.0f, t.imag(), 1e-6);
    ASSERT_NEAR(0.0f, t.real(), 1e-6);
  }
}

TEST(sadikov_d_crs_mult_seq, test_small_real_nonsquare_matrix) {
  matrix_CRS A;
  A.n = 4;
  A.m = 5;
  A.row_id = {0, 3, 6, 8, 10};
  A.col = {0, 3, 4, 1, 2, 3, 3, 4, 0, 4};
  A.value = {1, 2, 3, 2, 3, 1, -3, 2, 1, 3};
  matrix_CRS B;
  B.n = 5;
  B.m = 3;
  B.row_id = {0, 3, 4, 6, 6, 9};
  B.col = {0, 1, 2, 2, 0, 1, 0, 1, 2};
  B.value = {1, 3, 4, -1, 3, 1, 1, -1, -1};
  matrix_CRS C;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C));

  // Create Task
  CRSComplexMult_Sequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  ASSERT_EQ(taskSequential.pre_processing(), true);
  ASSERT_EQ(taskSequential.run(), true);
  ASSERT_EQ(taskSequential.post_processing(), true);

  // Check Result
  matrix_CRS C_expected;
  C_expected.n = 4;
  C_expected.m = 3;
  C_expected.row_id = {0, 2, 5, 8, 10};
  C_expected.col = {0, 2, 0, 1, 2, 0, 1, 2, 0, 2};
  C_expected.value = {4, 1, 9, 3, -2, 2, -2, -2, 4, 1};

  ASSERT_EQ(C.n, C_expected.n);
  ASSERT_EQ(C.m, C_expected.m);
  ASSERT_EQ(C.row_id, C_expected.row_id);
  ASSERT_EQ(C.col, C_expected.col);
  ASSERT_EQ(C.value.size(), C_expected.value.size());
  for (size_t i = 0; i < C_expected.value.size(); i++) {
    std::complex<double> t = C_expected.value[i] - C.value[i];
    ASSERT_NEAR(0.0f, t.imag(), 1e-6);
    ASSERT_NEAR(0.0f, t.real(), 1e-6);
  }
}

TEST(sadikov_d_crs_mult_seq, test_small_complex_nonsquare_matrix) {
  matrix_CRS A;
  A.n = 4;
  A.m = 4;
  A.row_id = {0, 2, 4, 5, 7};
  A.col = {0, 3, 1, 3, 0, 1, 3};
  A.value = {{1, 1}, {2, -1}, {0, -1}, {0, 1}, {1, 0}, {1, -1}, {1, -1}};
  matrix_CRS B;
  B.n = 4;
  B.m = 5;
  B.row_id = {0, 2, 5, 7, 9};
  B.col = {0, 4, 0, 1, 3, 2, 4, 1, 3};
  B.value = {{2, 1}, {0, -1}, {3, -1}, -1, 1, {0, 1}, {0, 1}, 1, -1};
  matrix_CRS C;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C));

  // Create Task
  CRSComplexMult_Sequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  ASSERT_EQ(taskSequential.pre_processing(), true);
  ASSERT_EQ(taskSequential.run(), true);
  ASSERT_EQ(taskSequential.post_processing(), true);

  // Check Result
  matrix_CRS C_expected;
  C_expected.n = 4;
  C_expected.m = 5;
  C_expected.row_id = {0, 4, 7, 9, 10};
  C_expected.col = {0, 1, 3, 4, 0, 1, 3, 0, 4, 0};
  C_expected.value = {{1, 3}, {2, -1}, {-2, 1}, {1, -1}, {-1, -3}, {0, 2}, {0, -2}, {2, 1}, {0, -1}, {2, -4}};

  ASSERT_EQ(C.n, C_expected.n);
  ASSERT_EQ(C.m, C_expected.m);
  ASSERT_EQ(C.row_id, C_expected.row_id);
  ASSERT_EQ(C.col, C_expected.col);
  ASSERT_EQ(C.value.size(), C_expected.value.size());
  for (size_t i = 0; i < C_expected.value.size(); i++) {
    std::complex<double> t = C_expected.value[i] - C.value[i];
    ASSERT_NEAR(0.0f, t.imag(), 1e-6);
    ASSERT_NEAR(0.0f, t.real(), 1e-6);
  }
}

TEST(sadikov_d_crs_mult_seq, test_diagonal_1_1_matrix) {
  matrix_CRS A;
  A.n = 25;
  A.m = 25;
  A.row_id = std::vector<int>(A.n + 1);
  std::iota(A.row_id.begin(), A.row_id.end(), 0);
  A.col = std::vector<int>(A.n);
  std::iota(A.col.begin(), A.col.end(), 0);
  A.value = std::vector<std::complex<double>>(A.n, std::complex<double>(1.0, 1.0));
  matrix_CRS B = A;
  matrix_CRS C;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C));

  // Create Task
  CRSComplexMult_Sequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  ASSERT_EQ(taskSequential.pre_processing(), true);
  ASSERT_EQ(taskSequential.run(), true);
  ASSERT_EQ(taskSequential.post_processing(), true);

  // Check Result
  matrix_CRS C_expected = A;
  for (auto &i : C_expected.value) {
    i = std::complex<double>(0.0, 2.0);
  }

  ASSERT_EQ(C.n, C_expected.n);
  ASSERT_EQ(C.m, C_expected.m);
  ASSERT_EQ(C.row_id, C_expected.row_id);
  ASSERT_EQ(C.col, C_expected.col);
  ASSERT_EQ(C.value.size(), C_expected.value.size());
  for (size_t i = 0; i < C_expected.value.size(); i++) {
    std::complex<double> t = C_expected.value[i] - C.value[i];
    ASSERT_NEAR(0.0f, t.imag(), 1e-6);
    ASSERT_NEAR(0.0f, t.real(), 1e-6);
  }
}

TEST(sadikov_d_crs_mult_seq, test_multiply_by_inverse_matrix) {
  matrix_CRS A;
  A.n = 4;
  A.m = 4;
  A.row_id = {0, 1, 3, 5, 8};
  A.col = {0, 1, 3, 0, 2, 0, 1, 3};
  A.value = {1, {0, 1}, {0, 1}, 1, {1, 1}, 2, 3, {2, -1}};
  matrix_CRS B;
  B.n = 4;
  B.m = 4;
  B.row_id = {0, 1, 4, 6, 9};
  B.col = {0, 0, 1, 3, 0, 2, 0, 1, 3};
  B.value = {1, {-1, 1}, {1.5, 0.5}, {0.5, -0.5}, {-0.5, 0.5}, {0.5, -0.5}, {1, -1}, {-1.5, -1.5}, {-0.5, 0.5}};
  matrix_CRS C;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&B));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&C));

  // Create Task
  CRSComplexMult_Sequential taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), true);
  ASSERT_EQ(taskSequential.pre_processing(), true);
  ASSERT_EQ(taskSequential.run(), true);
  ASSERT_EQ(taskSequential.post_processing(), true);

  // Check Result
  matrix_CRS C_expected;
  C_expected.n = 4;
  C_expected.m = 4;
  C_expected.row_id = {0, 1, 2, 3, 4};
  C_expected.col = {0, 1, 2, 3};
  C_expected.value = {1, 1, 1, 1};

  ASSERT_EQ(C.n, C_expected.n);
  ASSERT_EQ(C.m, C_expected.m);
  ASSERT_EQ(C.row_id, C_expected.row_id);
  ASSERT_EQ(C.col, C_expected.col);
  ASSERT_EQ(C.value.size(), C_expected.value.size());
  for (size_t i = 0; i < C_expected.value.size(); i++) {
    std::complex<double> t = C_expected.value[i] - C.value[i];
    ASSERT_NEAR(0.0f, t.imag(), 1e-6);
    ASSERT_NEAR(0.0f, t.real(), 1e-6);
  }
}
