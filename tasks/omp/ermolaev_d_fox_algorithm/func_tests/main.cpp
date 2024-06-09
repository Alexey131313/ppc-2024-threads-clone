#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "omp/ermolaev_d_fox_algorithm/include/ops_omp.hpp"

TEST(ermolaev_d_fox_algorithm_omp, Test_Matrix_Multiplication_Simple_56) {
  constexpr size_t matrix_size = 56;
  std::vector<double> A(matrix_size * matrix_size, 1.0);
  std::vector<double> B(matrix_size * matrix_size, 2.0);
  std::vector<double> C(matrix_size * matrix_size, 0.0);
  std::vector<double> expected(matrix_size * matrix_size, matrix_size * 2.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
  taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_size * matrix_size);

  FoxAlgorithmOMP test(taskDataSeq);
  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < matrix_size * matrix_size; i++) {
    ASSERT_DOUBLE_EQ(C[i], expected[i]);
  }
}

TEST(ermolaev_d_fox_algorithm_omp, Test_Matrix_Multiplication_Random_56) {
  constexpr size_t matrix_size = 56;
  std::mt19937 gen(1);
  const double tolerance = 1e-5;
  std::uniform_real_distribution<> dis(1.0, 6.0);
  std::vector<double> A(matrix_size * matrix_size);
  std::vector<double> B(matrix_size * matrix_size);
  std::vector<double> C(matrix_size * matrix_size, 0.0);
  std::vector<double> expected(matrix_size * matrix_size);

  for (size_t i = 0; i < matrix_size * matrix_size; ++i) {
    double random_value = dis(gen);
    A[i] = B[i] = random_value;
  }

  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      for (size_t k = 0; k < matrix_size; ++k) {
        expected[i * matrix_size + j] += A[i * matrix_size + k] * B[k * matrix_size + j];
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
  taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_size * matrix_size);

  FoxAlgorithmOMP test(taskDataSeq);
  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < matrix_size * matrix_size; ++i) {
    ASSERT_NEAR(C[i], expected[i], tolerance);
  }
}

TEST(ermolaev_d_fox_algorithm_omp, Test_Matrix_Multiplication_Simple_32) {
  constexpr size_t matrix_size = 32;
  std::vector<double> A(matrix_size * matrix_size, 1.0);
  std::vector<double> B(matrix_size * matrix_size, 2.0);
  std::vector<double> C(matrix_size * matrix_size, 0.0);
  std::vector<double> expected(matrix_size * matrix_size, matrix_size * 2.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
  taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_size * matrix_size);

  FoxAlgorithmOMP test(taskDataSeq);
  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < matrix_size * matrix_size; i++) {
    ASSERT_DOUBLE_EQ(C[i], expected[i]);
  }
}

TEST(ermolaev_d_fox_algorithm_omp, Test_Matrix_Multiplication_Random_32) {
  constexpr size_t matrix_size = 32;
  const double tolerance = 1e-5;
  std::mt19937 gen(1);
  std::uniform_real_distribution<> dis(1.0, 6.0);
  std::vector<double> A(matrix_size * matrix_size);
  std::vector<double> B(matrix_size * matrix_size);
  std::vector<double> C(matrix_size * matrix_size, 0.0);
  std::vector<double> expected(matrix_size * matrix_size);

  for (size_t i = 0; i < matrix_size * matrix_size; ++i) {
    double random_value = dis(gen);
    A[i] = B[i] = random_value;
  }

  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      for (size_t k = 0; k < matrix_size; ++k) {
        expected[i * matrix_size + j] += A[i * matrix_size + k] * B[k * matrix_size + j];
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
  taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_size * matrix_size);

  FoxAlgorithmOMP test(taskDataSeq);
  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < matrix_size * matrix_size; ++i) {
    ASSERT_NEAR(C[i], expected[i], tolerance);
  }
}

TEST(ermolaev_d_fox_algorithm_omp, Test_Matrix_Multiplication_Simple_64) {
  constexpr size_t matrix_size = 64;
  std::vector<double> A(matrix_size * matrix_size, 1.0);
  std::vector<double> B(matrix_size * matrix_size, 2.0);
  std::vector<double> C(matrix_size * matrix_size, 0.0);
  std::vector<double> expected(matrix_size * matrix_size, matrix_size * 2.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
  taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_size * matrix_size);

  FoxAlgorithmOMP test(taskDataSeq);
  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < matrix_size * matrix_size; i++) {
    ASSERT_DOUBLE_EQ(C[i], expected[i]);
  }
}

TEST(ermolaev_d_fox_algorithm_omp, Test_Matrix_Multiplication_Random_64) {
  constexpr size_t matrix_size = 64;
  const double tolerance = 1e-5;
  std::mt19937 gen(1);
  std::uniform_real_distribution<> dis(1.0, 6.0);
  std::vector<double> A(matrix_size * matrix_size);
  std::vector<double> B(matrix_size * matrix_size);
  std::vector<double> C(matrix_size * matrix_size, 0.0);
  std::vector<double> expected(matrix_size * matrix_size);

  for (size_t i = 0; i < matrix_size * matrix_size; ++i) {
    double random_value = dis(gen);
    A[i] = B[i] = random_value;
  }

  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      for (size_t k = 0; k < matrix_size; ++k) {
        expected[i * matrix_size + j] += A[i * matrix_size + k] * B[k * matrix_size + j];
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
  taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_size * matrix_size);

  FoxAlgorithmOMP test(taskDataSeq);
  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < matrix_size * matrix_size; ++i) {
    ASSERT_NEAR(C[i], expected[i], tolerance);
  }
}