// Copyright 2024 Shubin Mikhail

#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "seq/shubin_m_double_crs_mult/include/ops_seq.hpp"

TEST(shubin_m_double_crs_mult_seq, incorr_in_mats) {
  size_t mat_in_1_row = 2;
  size_t mat_in_1_col = 3;
  size_t mat_in_2_row = 4;
  size_t mat_in_2_col = 5;
  SparseMat_CRS mat_in_1(mat_in_1_row, mat_in_1_col);
  SparseMat_CRS mat_in_2(mat_in_2_row, mat_in_2_col);
  SparseMat_CRS mat_out(mat_in_1_row, mat_in_2_col);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_in_1));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_in_2));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_out));

  SparseMat_CRS_Mult_Seq crs_mat_mult_seq(taskDataSeq);
  ASSERT_FALSE(crs_mat_mult_seq.validation());
}

TEST(shubin_m_double_crs_mult_seq, mat_in_1_zero) {
  size_t mat_in_1_row = 3;
  size_t mat_in_1_col = 3;
  size_t mat_in_2_row = 3;
  size_t mat_in_2_col = 3;
  std::vector<double> z_mat(mat_in_1_row * mat_in_1_col, 0.0);
  std::vector<double> nz_mat(mat_in_2_row * mat_in_2_col);
  for (size_t i = 0; i < mat_in_2_row * mat_in_2_col; i++) {
    nz_mat[i] = (i % 3 == 0) ? (i) : (0.0);
  }

  SparseMat_CRS mat_in_1(z_mat, mat_in_1_row, mat_in_1_col);
  SparseMat_CRS mat_in_2(nz_mat, mat_in_2_row, mat_in_2_col);
  SparseMat_CRS mat_out(mat_in_1_row, mat_in_2_col);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_in_1));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_in_2));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_out));

  SparseMat_CRS_Mult_Seq crs_mat_mult_seq(taskDataSeq);
  ASSERT_TRUE(crs_mat_mult_seq.validation());
  ASSERT_TRUE(crs_mat_mult_seq.pre_processing());
  ASSERT_TRUE(crs_mat_mult_seq.run());
  ASSERT_TRUE(crs_mat_mult_seq.post_processing());

  std::vector<double> res = CRS_to_vector(mat_out);

  for (size_t i = 0; i < res.size(); i++) {
    EXPECT_DOUBLE_EQ(res[i], 0.0);
  }
}

TEST(shubin_m_double_crs_mult_seq, mat_in_2_zero) {
  size_t mat_in_1_row = 3;
  size_t mat_in_1_col = 3;
  size_t mat_in_2_row = 3;
  size_t mat_in_2_col = 3;
  std::vector<double> nz_mat(mat_in_1_row * mat_in_1_col);
  for (size_t i = 0; i < mat_in_1_row * mat_in_1_col; i++) {
    nz_mat[i] = (i % 3 == 0) ? (i + 1.0) : (0.0);
  }
  std::vector<double> z_mat(mat_in_2_row * mat_in_2_col, 0.0);

  SparseMat_CRS mat_in_1(nz_mat, mat_in_1_row, mat_in_1_col);
  SparseMat_CRS mat_in_2(z_mat, mat_in_2_row, mat_in_2_col);
  SparseMat_CRS mat_out(mat_in_1_row, mat_in_2_col);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_in_1));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_in_2));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_out));

  SparseMat_CRS_Mult_Seq crs_mat_mult_seq(taskDataSeq);
  ASSERT_TRUE(crs_mat_mult_seq.validation());
  ASSERT_TRUE(crs_mat_mult_seq.pre_processing());
  ASSERT_TRUE(crs_mat_mult_seq.run());
  ASSERT_TRUE(crs_mat_mult_seq.post_processing());

  std::vector<double> res = CRS_to_vector(mat_out);

  for (size_t i = 0; i < res.size(); i++) {
    EXPECT_DOUBLE_EQ(res[i], 0.0);
  }
}

TEST(shubin_m_double_crs_mult_seq, mat_in_1_ident) {
  size_t mat_in_1_row = 3;
  // size_t mat_in_1_col = 3;
  size_t mat_in_2_row = 3;
  size_t mat_in_2_col = 3;
  std::vector<double> nz_mat(mat_in_2_row * mat_in_2_col);
  for (size_t i = 0; i < mat_in_2_row * mat_in_2_col; i++) {
    nz_mat[i] = (i % 3 == 0) ? (i + 1.0) : (0.0);
  }

  SparseMat_CRS mat_in_1 = ident_CRS_mat(mat_in_1_row);
  SparseMat_CRS mat_in_2(nz_mat, mat_in_2_row, mat_in_2_col);
  SparseMat_CRS mat_out(mat_in_1_row, mat_in_2_col);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_in_1));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_in_2));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_out));

  SparseMat_CRS_Mult_Seq crs_mat_mult_seq(taskDataSeq);
  ASSERT_TRUE(crs_mat_mult_seq.validation());
  ASSERT_TRUE(crs_mat_mult_seq.pre_processing());
  ASSERT_TRUE(crs_mat_mult_seq.run());
  ASSERT_TRUE(crs_mat_mult_seq.post_processing());

  std::vector<double> res = CRS_to_vector(mat_out);

  for (size_t i = 0; i < res.size(); i++) {
    EXPECT_DOUBLE_EQ(res[i], (i % 3 == 0) ? (i + 1.0) : (0.0));
  }
}

TEST(shubin_m_double_crs_mult_seq, mat_in_2_ident) {
  size_t mat_in_1_row = 3;
  size_t mat_in_1_col = 3;
  // size_t mat_in_2_row = 3;
  size_t mat_in_2_col = 3;
  std::vector<double> nz_mat(mat_in_1_row * mat_in_1_col);
  for (size_t i = 0; i < mat_in_1_row * mat_in_1_col; i++) {
    nz_mat[i] = (i % 3 == 0) ? (i + 1.0) : (0.0);
  }

  SparseMat_CRS mat_in_1(nz_mat, mat_in_1_row, mat_in_1_col);
  SparseMat_CRS mat_in_2 = ident_CRS_mat(mat_in_2_col);
  SparseMat_CRS mat_out(mat_in_1_row, mat_in_2_col);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_in_1));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_in_2));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_out));

  SparseMat_CRS_Mult_Seq crs_mat_mult_seq(taskDataSeq);
  ASSERT_TRUE(crs_mat_mult_seq.validation());
  ASSERT_TRUE(crs_mat_mult_seq.pre_processing());
  ASSERT_TRUE(crs_mat_mult_seq.run());
  ASSERT_TRUE(crs_mat_mult_seq.post_processing());

  std::vector<double> res = CRS_to_vector(mat_out);

  for (size_t i = 0; i < res.size(); i++) {
    EXPECT_DOUBLE_EQ(res[i], (i % 3 == 0) ? (i + 1.0) : (0.0));
  }
}

TEST(shubin_m_double_crs_mult_seq, random_small_dim_mat_mult) {
  size_t mat_in_1_row = 3;
  size_t mat_in_1_col = 3;
  size_t mat_in_2_row = 3;
  size_t mat_in_2_col = 3;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> val_dist(-100.0, 100.0);
  std::uniform_int_distribution<size_t> ind_dist_1(0, mat_in_1_row * mat_in_1_col - 1);
  std::uniform_int_distribution<size_t> ind_dist_2(0, mat_in_2_row * mat_in_2_col - 1);

  std::vector<double> A(mat_in_1_row * mat_in_1_col, 0.0);
  for (size_t i = 0; i < static_cast<size_t>(0.1 * mat_in_1_row * mat_in_1_col); i++) {
    A[ind_dist_1(gen)] = val_dist(gen);
  }
  std::vector<double> B(mat_in_2_row * mat_in_2_col, 0.0);
  for (size_t i = 0; i < static_cast<size_t>(0.1 * mat_in_2_row * mat_in_2_col); i++) {
    B[ind_dist_2(gen)] = val_dist(gen);
  }
  std::vector<double> C(mat_in_1_row * mat_in_2_col, 0.0);
  for (size_t i = 0; i < mat_in_1_row; i++) {
    for (size_t j = 0; j < mat_in_2_col; j++) {
      for (size_t k = 0; k < mat_in_1_col; k++) {
        C[i * mat_in_1_col + j] += A[i * mat_in_1_col + k] * B[k * mat_in_2_col + j];
      }
    }
  }

  SparseMat_CRS mat_in_1(A, mat_in_1_row, mat_in_1_col);
  SparseMat_CRS mat_in_2(B, mat_in_2_row, mat_in_2_col);
  SparseMat_CRS mat_out(mat_in_1_row, mat_in_2_col);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_in_1));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_in_2));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&mat_out));

  SparseMat_CRS_Mult_Seq crs_mat_mult_seq(taskDataSeq);
  ASSERT_TRUE(crs_mat_mult_seq.validation());
  ASSERT_TRUE(crs_mat_mult_seq.pre_processing());
  ASSERT_TRUE(crs_mat_mult_seq.run());
  ASSERT_TRUE(crs_mat_mult_seq.post_processing());

  std::vector<double> res = CRS_to_vector(mat_out);

  for (size_t i = 0; i < res.size(); i++) {
    EXPECT_DOUBLE_EQ(res[i], C[i]);
  }
}
