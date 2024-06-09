// Copyright 2024 Lysanova Julia
#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "seq/lysanova_j_matr_fox/include/ops_seq.hpp"

TEST(lysanova_j_matr_fox_seq, AE) {
  size_t n = 2;
  std::vector<double> A{42.0, 54.0, -13.0, 0.0};
  std::vector<double> B{1.0, 0.0, 0.0, 1.0};
  std::vector<double> C(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  // Create Task
  LysanovaTaskSequential lysanovaTaskSequential(taskDataSeq);
  ASSERT_EQ(lysanovaTaskSequential.validation(), true);
  lysanovaTaskSequential.pre_processing();
  lysanovaTaskSequential.run();
  lysanovaTaskSequential.post_processing();
  for (size_t i = 0; i < n * n; i++) {
    EXPECT_DOUBLE_EQ(C[i], A[i]);
  }
}

TEST(lysanova_j_matr_fox_seq, AEsd) {
  size_t n = 2;
  std::vector<double> A{42.0, 54.0, -13.0, 0.0};
  std::vector<double> B{0.0, 1.0, 1.0, 0.0};
  std::vector<double> C(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  // Create Task
  LysanovaTaskSequential lysanovaTaskSequential(taskDataSeq);
  ASSERT_EQ(lysanovaTaskSequential.validation(), true);
  lysanovaTaskSequential.pre_processing();
  lysanovaTaskSequential.run();
  lysanovaTaskSequential.post_processing();
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      EXPECT_DOUBLE_EQ(C[i * n + j], A[i * n + (n - j - 1)]);
    }
  }
}

TEST(lysanova_j_matr_fox_seq, EA) {
  size_t n = 2;
  std::vector<double> A{1.0, 0.0, 0.0, 1.0};
  std::vector<double> B{42.0, 54.0, -13.0, 0.0};
  std::vector<double> C(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  // Create Task
  LysanovaTaskSequential lysanovaTaskSequential(taskDataSeq);
  ASSERT_EQ(lysanovaTaskSequential.validation(), true);
  lysanovaTaskSequential.pre_processing();
  lysanovaTaskSequential.run();
  lysanovaTaskSequential.post_processing();
  for (size_t i = 0; i < n * n; i++) {
    EXPECT_DOUBLE_EQ(C[i], B[i]);
  }
}

TEST(lysanova_j_matr_fox_seq, EsdA) {
  size_t n = 2;
  std::vector<double> A{0.0, 1.0, 1.0, 0.0};
  std::vector<double> B{42.0, 54.0, -13.0, 0.0};
  std::vector<double> C(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  // Create Task
  LysanovaTaskSequential lysanovaTaskSequential(taskDataSeq);
  ASSERT_EQ(lysanovaTaskSequential.validation(), true);
  lysanovaTaskSequential.pre_processing();
  lysanovaTaskSequential.run();
  lysanovaTaskSequential.post_processing();
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      EXPECT_DOUBLE_EQ(C[i * n + j], B[((n - i - 1)) * n + j]);
    }
  }
}

TEST(lysanova_j_matr_fox_seq, kAE) {
  size_t n = 2;
  double k = 42.0;
  std::vector<double> A{42.0, 54.0, -13.0, 0.0};
  std::vector<double> B{k, 0.0, 0.0, k};
  std::vector<double> C(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(B.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  // Create Task
  LysanovaTaskSequential lysanovaTaskSequential(taskDataSeq);
  ASSERT_EQ(lysanovaTaskSequential.validation(), true);
  lysanovaTaskSequential.pre_processing();
  lysanovaTaskSequential.run();
  lysanovaTaskSequential.post_processing();
  for (size_t i = 0; i < n * n; i++) {
    EXPECT_DOUBLE_EQ(C[i], k * A[i]);
  }
}
