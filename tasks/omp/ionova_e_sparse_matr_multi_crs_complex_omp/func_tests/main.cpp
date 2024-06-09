// Copyright 2024 Ionova Ekatetina
#include <gtest/gtest.h>

#include <vector>

#include "omp/ionova_e_sparse_matr_multi_crs_complex_omp/include/ops_seq.hpp"

TEST(ionova_e_sparse_matr_multi_crs_complex_omp, sizes_correct) {
  size_t n1 = 4;
  size_t m1 = 4;
  size_t n2 = 4;
  size_t m2 = 4;

  // Create data
  std::vector<Complex> in1(n1 * m1);

  std::vector<Complex> in2(n2 * m2);

  std::vector<Complex> out(n1 * m2);

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
  SparseMatrixComplexMultiSequentialOmp sparseMatrixComplexMultiSequentialOmp(taskDataSeq);
  ASSERT_EQ(sparseMatrixComplexMultiSequentialOmp.validation(), true);

  // Create data
  std::vector<Complex> in3(n1 * m1);

  std::vector<Complex> in4(n2 * m2);

  std::vector<Complex> out2(n1 * m2);

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

  SparseMatrixComplexMultiParallelOmp sparseMatrixComplexMultiParallelOmp(taskDataParallel);
  ASSERT_EQ(sparseMatrixComplexMultiParallelOmp.validation(), true);
}

TEST(ionova_e_sparse_matr_multi_crs_complex_omp, sizes_incorrect) {
  size_t n1 = 4;
  size_t m1 = 5;
  size_t n2 = 3;
  size_t m2 = 4;

  // Create data
  std::vector<Complex> in1(n1 * m1);

  std::vector<Complex> in2(n2 * m2);

  std::vector<Complex> out(n1 * m2);

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
  SparseMatrixComplexMultiSequentialOmp sparseMatrixComplexMultiSequentialOmp(taskDataSeq);
  ASSERT_FALSE(sparseMatrixComplexMultiSequentialOmp.validation());

  // Create data
  std::vector<Complex> in3(n1 * m1);

  std::vector<Complex> in4(n2 * m2);

  std::vector<Complex> out2(n1 * m2);

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

  SparseMatrixComplexMultiParallelOmp sparseMatrixComplexMultiParallelOmp(taskDataParallel);
  ASSERT_FALSE(sparseMatrixComplexMultiParallelOmp.validation());
}

TEST(ionova_e_sparse_matr_multi_crs_complex_omp, zero_matrix) {
  size_t n1 = 3;
  size_t m1 = 3;
  size_t n2 = 3;
  size_t m2 = 3;
  // Create data
  Complex compNull = {0, 0};
  std::vector<Complex> in1{{3, 2}, {0, 0}, {0, 0}, {0, 0}, {1, -3}, {0, 0}, {0, 0}, {0, 0}, {-4, 1}};
  std::vector<Complex> in2{compNull, compNull, compNull, compNull, compNull, compNull, compNull, compNull, compNull};
  std::vector<Complex> out(n1 * m2);

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
  SparseMatrixComplexMultiSequentialOmp sparseMatrixComplexMultiSequentialOmp(taskDataSeq);
  sparseMatrixComplexMultiSequentialOmp.validation();
  sparseMatrixComplexMultiSequentialOmp.pre_processing();
  sparseMatrixComplexMultiSequentialOmp.run();
  sparseMatrixComplexMultiSequentialOmp.post_processing();

  size_t k = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i].real == 0.0 && out[i].imag == 0.0) {
      k++;
    }
  }

  ASSERT_EQ(k, n1 * m2);

  // Create data
  std::vector<Complex> out2(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataParallel->inputs_count.emplace_back(n1);
  taskDataParallel->inputs_count.emplace_back(m1);
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataParallel->inputs_count.emplace_back(n2);
  taskDataParallel->inputs_count.emplace_back(m2);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataParallel->outputs_count.emplace_back(n1);
  taskDataParallel->outputs_count.emplace_back(m2);

  SparseMatrixComplexMultiParallelOmp sparseMatrixComplexMultiParallelOmp(taskDataSeq);
  sparseMatrixComplexMultiParallelOmp.validation();
  sparseMatrixComplexMultiParallelOmp.pre_processing();
  sparseMatrixComplexMultiParallelOmp.run();
  sparseMatrixComplexMultiParallelOmp.post_processing();

  size_t ch = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i].real == 0.0 && out[i].imag == 0.0) {
      ch++;
    }
  }

  ASSERT_EQ(ch, n1 * m2);
}

TEST(ionova_e_sparse_matr_multi_crs_complex_omp, test_multy) {
  size_t n1 = 3;
  size_t m1 = 3;
  size_t n2 = 3;
  size_t m2 = 3;

  // Create data
  std::vector<Complex> in1{{3, 2}, {0, 0}, {0, 0}, {0, 0}, {1, -3}, {0, 0}, {0, 0}, {0, 0}, {-4, 1}};
  std::vector<Complex> in2{{0, 0}, {2, -1}, {0, 0}, {0, 0}, {0, 0}, {-5, 2}, {0, 0}, {2, 1}, {0, 0}};
  std::vector<Complex> out(n1 * m2);
  std::vector<Complex> test{{0, 0}, {8, 1}, {0, 0}, {0, 0}, {0, 0}, {1, 17}, {0, 0}, {-9, -2}, {0, 0}};

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
  SparseMatrixComplexMultiSequentialOmp sparseMatrixComplexMultiSequentialOmp(taskDataSeq);
  sparseMatrixComplexMultiSequentialOmp.validation();
  sparseMatrixComplexMultiSequentialOmp.pre_processing();
  sparseMatrixComplexMultiSequentialOmp.run();
  sparseMatrixComplexMultiSequentialOmp.post_processing();

  size_t k = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i].real == test[i].real && out[i].imag == test[i].imag) {
      k++;
    }
  }

  ASSERT_EQ(k, n1 * m2);

  // Create data
  std::vector<Complex> out2(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataParallel->inputs_count.emplace_back(n1);
  taskDataParallel->inputs_count.emplace_back(m1);
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataParallel->inputs_count.emplace_back(n2);
  taskDataParallel->inputs_count.emplace_back(m2);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataParallel->outputs_count.emplace_back(n1);
  taskDataParallel->outputs_count.emplace_back(m2);

  SparseMatrixComplexMultiParallelOmp sparseMatrixComplexMultiParallelOmp(taskDataSeq);
  sparseMatrixComplexMultiParallelOmp.validation();
  sparseMatrixComplexMultiParallelOmp.pre_processing();
  sparseMatrixComplexMultiParallelOmp.run();
  sparseMatrixComplexMultiParallelOmp.post_processing();

  size_t ch = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i].real == test[i].real && out[i].imag == test[i].imag) {
      ch++;
    }
  }

  ASSERT_EQ(ch, n1 * m2);
}

TEST(ionova_e_sparse_matr_multi_crs_complex_omp, inverse_matrix) {
  size_t n1 = 3;
  size_t m1 = 3;
  size_t n2 = 3;
  size_t m2 = 3;

  // Create data
  std::vector<Complex> in1{{2, -1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 2}, {0, 0}, {2, 1}, {0, 0}};
  std::vector<Complex> in2{{0.4, 0.2}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0.4, -0.2}, {0, 0}, {0, -0.5}, {0, 0}};
  std::vector<Complex> out(n1 * m2);
  std::vector<Complex> test{{1, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}};

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
  SparseMatrixComplexMultiSequentialOmp sparseMatrixComplexMultiSequentialOmp(taskDataSeq);
  sparseMatrixComplexMultiSequentialOmp.validation();
  sparseMatrixComplexMultiSequentialOmp.pre_processing();
  sparseMatrixComplexMultiSequentialOmp.run();
  sparseMatrixComplexMultiSequentialOmp.post_processing();

  size_t k = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i].real == test[i].real && out[i].imag == test[i].imag) {
      k++;
    }
  }

  ASSERT_EQ(k, n1 * m2);

  // Create data
  std::vector<Complex> out2(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataParallel->inputs_count.emplace_back(n1);
  taskDataParallel->inputs_count.emplace_back(m1);
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataParallel->inputs_count.emplace_back(n2);
  taskDataParallel->inputs_count.emplace_back(m2);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataParallel->outputs_count.emplace_back(n1);
  taskDataParallel->outputs_count.emplace_back(m2);

  SparseMatrixComplexMultiParallelOmp sparseMatrixComplexMultiParallelOmp(taskDataSeq);
  sparseMatrixComplexMultiParallelOmp.validation();
  sparseMatrixComplexMultiParallelOmp.pre_processing();
  sparseMatrixComplexMultiParallelOmp.run();
  sparseMatrixComplexMultiParallelOmp.post_processing();

  size_t ch = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i].real == test[i].real && out[i].imag == test[i].imag) {
      ch++;
    }
  }

  ASSERT_EQ(ch, n1 * m2);
}
