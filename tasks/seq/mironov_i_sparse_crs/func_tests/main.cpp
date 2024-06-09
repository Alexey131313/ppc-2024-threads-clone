// Copyright 2024 Mironov Ilya
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "seq/mironov_i_sparse_crs/include/ops_seq.hpp"

TEST(mironov_i_sparse_crs_seq, Test1Static) {
  std::vector<double> C(3 * 3, 0.0);
  std::vector<double> A = {2, 0, 2, 0, 0, 1, 0, 2, 0, 0, 3, 0};
  std::vector<double> B = {2, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 3};
  std::vector<double> res = {4, 2, 4, 0, 0, 6, 0, 3, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(3);
  taskDataSeq->inputs_count.emplace_back(A.size() / 3);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(4);
  taskDataSeq->inputs_count.emplace_back(A.size() / 4);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  // Create Task
  MironovISequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  for (size_t i = 0; i < C.size(); i++) {
    EXPECT_DOUBLE_EQ(res[i], C[i]);
  }
}

TEST(mironov_i_sparse_crs_seq, Test2Static) {
  std::vector<double> C(3 * 3, 0.0);
  std::vector<double> A = {2, 0, 2, 0, 0, 1, 0, 2, -2, 0, 3, 0};
  std::vector<double> B = {2, 0, 0, 0, -3, 0, 0, 1, 4, 1, 2, 3};
  std::vector<double> res = {4, 2, 8, 2, 1, 6, -4, 3, 12};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(3);
  taskDataSeq->inputs_count.emplace_back(A.size() / 3);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(4);
  taskDataSeq->inputs_count.emplace_back(A.size() / 4);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  // Create Task
  MironovISequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  for (size_t i = 0; i < C.size(); i++) {
    EXPECT_DOUBLE_EQ(res[i], C[i]);
  }
}

TEST(mironov_i_sparse_crs_seq, TestEdin34) {
  std::vector<double> A = {2, 0, 2, 0, 0, 1, 0, 2, -2, 0, 3, 0};
  int n = 3;
  int m = A.size() / n;
  std::vector<double> B(m * m);
  MironovISequential::genrateEdMatrix(B.data(), m);
  std::vector<double> C(n * m, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  // Create Task
  MironovISequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  for (size_t i = 0; i < C.size(); i++) {
    EXPECT_DOUBLE_EQ(A[i], C[i]);
  }
}

TEST(mironov_i_sparse_crs_seq, TestEdin43) {
  std::vector<double> A = {2, 0, 2, 0, 0, 1, 0, 2, -2, 0, 3, 0};
  int n = 4;
  int m = A.size() / n;
  std::vector<double> B(m * m);
  MironovISequential::genrateEdMatrix(B.data(), m);
  std::vector<double> C(n * m, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  // Create Task
  MironovISequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  for (size_t i = 0; i < C.size(); i++) {
    EXPECT_DOUBLE_EQ(A[i], C[i]);
  }
}

TEST(mironov_i_sparse_crs_seq, TestRandom) {
  int n = 15;
  int m = 15;
  int k = 15;
  double ro = 0.3;
  std::vector<double> A(n * m, 0.0);
  MironovISequential::genrateSparseMatrix(A.data(), A.size(), ro);
  std::vector<double> B(m * k, 0.0);
  MironovISequential::genrateSparseMatrix(B.data(), B.size(), ro);
  std::vector<double> C(n * k, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(k);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  // Create Task
  MironovISequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
}
