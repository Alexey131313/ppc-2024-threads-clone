// Copyright 2023 Simonyan Suren
#include <gtest/gtest.h>

#include <vector>

#include "tbb/simonyan_s_sparse_matr_multi_ccs_tbb/include/ccs_mat_multy.hpp"

TEST(simonyan_s_sparse_matr_multi_ccs_tbb, test_sizes) {
  size_t n1 = 4;
  size_t m1 = 5;
  size_t n2 = 3;
  size_t m2 = 4;

  // Create data
  std::vector<double> in1(n1 * m1);

  std::vector<double> in2(n2 * m2);

  std::vector<double> out(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(n1);
  taskDataSeq->inputs_count.emplace_back(m1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(n2);
  taskDataSeq->inputs_count.emplace_back(m2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n1);
  taskDataSeq->outputs_count.emplace_back(m2);

  // Create Task
  SparseTBBMatrixMultiSequential sparseTBBMatrixMultiSequential(taskDataSeq);
  ASSERT_FALSE(sparseTBBMatrixMultiSequential.validation());

  // Create data
  std::vector<double> in3(n1 * m1);

  std::vector<double> in4(n2 * m2);

  std::vector<double> out2(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataParallel->inputs_count.emplace_back(n1);
  taskDataParallel->inputs_count.emplace_back(m1);
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in4.data()));
  taskDataParallel->inputs_count.emplace_back(n2);
  taskDataParallel->inputs_count.emplace_back(m2);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataParallel->outputs_count.emplace_back(n1);
  taskDataParallel->outputs_count.emplace_back(m2);
  // Create Task
  SparseTBBMatrixMultiParallel sparseTBBMatrixMultiParallel(taskDataParallel);
  ASSERT_FALSE(sparseTBBMatrixMultiParallel.validation());
}

TEST(simonyan_s_sparse_matr_multi_ccs_tbb, test_sizes2) {
  size_t n1 = 4;
  size_t m1 = 4;
  size_t n2 = 4;
  size_t m2 = 4;

  // Create data
  std::vector<double> in1(n1 * m1);

  std::vector<double> in2(n2 * m2);

  std::vector<double> out(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(n1);
  taskDataSeq->inputs_count.emplace_back(m1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(n2);
  taskDataSeq->inputs_count.emplace_back(m2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n1);
  taskDataSeq->outputs_count.emplace_back(m2);

  // Create Task
  SparseTBBMatrixMultiSequential sparseTBBMatrixMultiSequential(taskDataSeq);
  ASSERT_EQ(sparseTBBMatrixMultiSequential.validation(), true);

  // Create data
  std::vector<double> in3(n1 * m1);

  std::vector<double> in4(n2 * m2);

  std::vector<double> out2(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataParallel->inputs_count.emplace_back(n1);
  taskDataParallel->inputs_count.emplace_back(m1);
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in4.data()));
  taskDataParallel->inputs_count.emplace_back(n2);
  taskDataParallel->inputs_count.emplace_back(m2);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataParallel->outputs_count.emplace_back(n1);
  taskDataParallel->outputs_count.emplace_back(m2);
  // Create Task
  SparseTBBMatrixMultiParallel sparseTBBMatrixMultiParallel(taskDataParallel);
  ASSERT_EQ(sparseTBBMatrixMultiParallel.validation(), true);
}

TEST(simonyan_s_sparse_matr_multi_ccs_tbb, test_multy_correct) {
  size_t n1 = 4;
  size_t m1 = 4;
  size_t n2 = 4;
  size_t m2 = 4;

  // Create data
  std::vector<double> in1{5, 0, 0, 0, 0, 0, 5, 0, 0, 1, 0, 0, 8, 0, 6, 0};
  std::vector<double> in2{5, 0, 0, 8, 0, 0, 1, 0, 0, 5, 0, 6, 0, 0, 0, 0};
  std::vector<double> out(n1 * m2);
  std::vector<double> test{25, 0, 0, 40, 0, 25, 0, 30, 0, 0, 1, 0, 40, 30, 0, 100};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(n1);
  taskDataSeq->inputs_count.emplace_back(m1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(n2);
  taskDataSeq->inputs_count.emplace_back(m2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n1);
  taskDataSeq->outputs_count.emplace_back(m2);

  // Create Task
  SparseTBBMatrixMultiSequential sparseTBBMatrixMultiSequential(taskDataSeq);
  sparseTBBMatrixMultiSequential.validation();
  sparseTBBMatrixMultiSequential.pre_processing();
  sparseTBBMatrixMultiSequential.run();
  sparseTBBMatrixMultiSequential.post_processing();

  size_t k = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == test[i]) {
      k++;
    }
  }

  ASSERT_EQ(k, n1 * m2);
  // Create data
  std::vector<double> in3{5, 0, 0, 0, 0, 0, 5, 0, 0, 1, 0, 0, 8, 0, 6, 0};
  std::vector<double> in4{5, 0, 0, 8, 0, 0, 1, 0, 0, 5, 0, 6, 0, 0, 0, 0};
  std::vector<double> out2(n1 * m2);
  std::vector<double> test2{25, 0, 0, 40, 0, 25, 0, 30, 0, 0, 1, 0, 40, 30, 0, 100};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataParallel->inputs_count.emplace_back(n1);
  taskDataParallel->inputs_count.emplace_back(m1);
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in4.data()));
  taskDataParallel->inputs_count.emplace_back(n2);
  taskDataParallel->inputs_count.emplace_back(m2);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataParallel->outputs_count.emplace_back(n1);
  taskDataParallel->outputs_count.emplace_back(m2);

  // Create Task
  SparseTBBMatrixMultiParallel sparseTBBMatrixMultiParallel(taskDataSeq);
  sparseTBBMatrixMultiParallel.validation();
  sparseTBBMatrixMultiParallel.pre_processing();
  sparseTBBMatrixMultiParallel.run();
  sparseTBBMatrixMultiParallel.post_processing();

  size_t ch = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == test[i]) {
      ch++;
    }
  }

  ASSERT_EQ(ch, n1 * m2);
}

TEST(simonyan_s_sparse_matr_multi_ccs_tbb, inverse_matrix) {
  size_t n1 = 4;
  size_t m1 = 4;
  size_t n2 = 4;
  size_t m2 = 4;

  // Create data
  std::vector<double> in1{0, 5, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 1, 0, 0, 0};
  std::vector<double> in2{0, 0, 0, 1, 0.2, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 1, 0};
  std::vector<double> out(n1 * m2);
  std::vector<double> identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(n1);
  taskDataSeq->inputs_count.emplace_back(m1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(n2);
  taskDataSeq->inputs_count.emplace_back(m2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n1);
  taskDataSeq->outputs_count.emplace_back(m2);

  // Create Task
  SparseTBBMatrixMultiSequential sparseTBBMatrixMultiSequential(taskDataSeq);
  sparseTBBMatrixMultiSequential.validation();
  sparseTBBMatrixMultiSequential.pre_processing();
  sparseTBBMatrixMultiSequential.run();
  sparseTBBMatrixMultiSequential.post_processing();

  size_t k = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == identity[i]) {
      k++;
    }
  }

  ASSERT_EQ(k, n1 * m2);
  // Create data
  std::vector<double> in3{0, 5, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 1, 0, 0, 0};
  std::vector<double> in4{0, 0, 0, 1, 0.2, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 1, 0};
  std::vector<double> out2(n1 * m2);
  std::vector<double> identity2{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataParallel->inputs_count.emplace_back(n1);
  taskDataParallel->inputs_count.emplace_back(m1);
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in4.data()));
  taskDataParallel->inputs_count.emplace_back(n2);
  taskDataParallel->inputs_count.emplace_back(m2);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataParallel->outputs_count.emplace_back(n1);
  taskDataParallel->outputs_count.emplace_back(m2);

  // Create Task
  SparseTBBMatrixMultiParallel sparseTBBMatrixMultiParallel(taskDataSeq);
  sparseTBBMatrixMultiParallel.validation();
  sparseTBBMatrixMultiParallel.pre_processing();
  sparseTBBMatrixMultiParallel.run();
  sparseTBBMatrixMultiParallel.post_processing();

  size_t ch = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == identity2[i]) {
      ch++;
    }
  }

  ASSERT_EQ(ch, n1 * m2);
}

TEST(simonyan_s_sparse_matr_multi_ccs_tbb, zero_matrix) {
  size_t n1 = 3;
  size_t m1 = 3;
  size_t n2 = 3;
  size_t m2 = 3;
  // Create data
  std::vector<double> in1{1, -2, 1, 2, 1, -1, 3, 2, -2};

  std::vector<double> in2(n2 * m2, 0);

  std::vector<double> out(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(n1);
  taskDataSeq->inputs_count.emplace_back(m1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(n2);
  taskDataSeq->inputs_count.emplace_back(m2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n1);
  taskDataSeq->outputs_count.emplace_back(m2);

  // Create Task
  SparseTBBMatrixMultiSequential sparseTBBMatrixMultiSequential(taskDataSeq);
  sparseTBBMatrixMultiSequential.validation();
  sparseTBBMatrixMultiSequential.pre_processing();
  sparseTBBMatrixMultiSequential.run();
  sparseTBBMatrixMultiSequential.post_processing();

  size_t k = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == 0.0) {
      k++;
    }
  }

  ASSERT_EQ(k, n1 * m2);
  // Create data
  std::vector<double> in3{1, -2, 1, 2, 1, -1, 3, 2, -2};
  ;
  std::vector<double> in4(n2 * m2, 0);
  std::vector<double> out2(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataParallel->inputs_count.emplace_back(n1);
  taskDataParallel->inputs_count.emplace_back(m1);
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in4.data()));
  taskDataParallel->inputs_count.emplace_back(n2);
  taskDataParallel->inputs_count.emplace_back(m2);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataParallel->outputs_count.emplace_back(n1);
  taskDataParallel->outputs_count.emplace_back(m2);

  // Create Task
  SparseTBBMatrixMultiParallel sparseTBBMatrixMultiParallel(taskDataSeq);
  sparseTBBMatrixMultiParallel.validation();
  sparseTBBMatrixMultiParallel.pre_processing();
  sparseTBBMatrixMultiParallel.run();
  sparseTBBMatrixMultiParallel.post_processing();

  size_t ch = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == 0.0) {
      ch++;
    }
  }

  ASSERT_EQ(ch, n1 * m2);
}