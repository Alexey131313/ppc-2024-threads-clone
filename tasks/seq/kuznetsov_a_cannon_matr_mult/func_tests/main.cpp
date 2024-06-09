// Copyright 2023 Kuznetsov Artem
#include <gtest/gtest.h>

#include <vector>

#include "seq/kuznetsov_a_cannon_matr_mult/include/ops_seq.hpp"

TEST(Kuznetsov_a_cannon_matr_mult_seq_func_tests, mult_3x3) {
  // Create data
  size_t size = 3;
  size_t block = 2;

  std::vector<double> inputMatrOne{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> inputMatrTwo{9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<double> outputMatr(size * size, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  // Add matrOne
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrOne.data()));
  taskDataSeq->inputs_count.emplace_back(inputMatrOne.size());

  // Add matrTwo
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrTwo.data()));
  taskDataSeq->inputs_count.emplace_back(inputMatrTwo.size());

  // Add size
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));

  // Add block
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block));

  // Add out matr
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputMatr.data()));
  taskDataSeq->outputs_count.emplace_back(outputMatr.size());

  auto resSeq = KuznetsovArtyomSeq::multMatrSquare(inputMatrOne, inputMatrTwo, size);

  // Create Task
  KuznetsovArtyomSeq::KuznetsovCannonMatrMultSeq testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  auto resSize = resSeq.size();

  for (size_t i = 0; i < resSize; ++i) {
    ASSERT_TRUE(KuznetsovArtyomSeq::isEqual(resSeq[i], outputMatr[i]));
  }
}

TEST(Kuznetsov_a_cannon_matr_mult_seq_func_tests, mult_4x4) {
  // Create data
  size_t size = 4;
  size_t block = 2;

  std::vector<double> inputMatrOne{2, 3, 4, 5, 9, 8, 7, 6, 5, 4, 2, 3, 8, 7, 3, 4};
  std::vector<double> inputMatrTwo{3, 5, 7, 6, 2, 7, 6, 3, 7, 5, 3, 2, 4, 3, 2, 5};
  std::vector<double> outputMatr(size * size, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  // Add matrOne
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrOne.data()));
  taskDataSeq->inputs_count.emplace_back(inputMatrOne.size());

  // Add matrTwo
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrTwo.data()));
  taskDataSeq->inputs_count.emplace_back(inputMatrTwo.size());

  // Add size
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));

  // Add block
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block));

  // Add out matr
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputMatr.data()));
  taskDataSeq->outputs_count.emplace_back(outputMatr.size());

  auto resSeq = KuznetsovArtyomSeq::multMatrSquare(inputMatrOne, inputMatrTwo, size);

  // Create Task
  KuznetsovArtyomSeq::KuznetsovCannonMatrMultSeq testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  auto resSize = resSeq.size();

  for (size_t i = 0; i < resSize; ++i) {
    ASSERT_TRUE(KuznetsovArtyomSeq::isEqual(resSeq[i], outputMatr[i]));
  }
}

TEST(Kuznetsov_a_cannon_matr_mult_seq_func_tests, mult_5x5) {
  // Create data
  size_t size = 5;
  size_t block = 3;

  std::vector<double> inputMatrOne{-17, -11, 9,  -21, -17, -18, 8, 4,   1,   -23, 19, -11, -9,
                                   -15, -18, 14, -8,  13,  -25, 7, -23, -11, 6,   12, -2};
  std::vector<double> inputMatrTwo{20, 19,  -4,  20, 10, -20, -14, 7,   9,  19, -20, -6, 17,
                                   1,  -19, -23, 7,  2,  13,  -14, -12, 18, 17, 2,   -18};
  std::vector<double> outputMatr(size * size, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  // Add matrOne
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrOne.data()));
  taskDataSeq->inputs_count.emplace_back(inputMatrOne.size());

  // Add matrTwo
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrTwo.data()));
  taskDataSeq->inputs_count.emplace_back(inputMatrTwo.size());

  // Add size
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));

  // Add block
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block));

  // Add out matr
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputMatr.data()));
  taskDataSeq->outputs_count.emplace_back(outputMatr.size());

  auto resSeq = KuznetsovArtyomSeq::multMatrSquare(inputMatrOne, inputMatrTwo, size);

  // Create Task
  KuznetsovArtyomSeq::KuznetsovCannonMatrMultSeq testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  auto resSize = resSeq.size();

  for (size_t i = 0; i < resSize; ++i) {
    ASSERT_TRUE(KuznetsovArtyomSeq::isEqual(resSeq[i], outputMatr[i]));
  }
}

TEST(Kuznetsov_a_cannon_matr_mult_seq_func_tests, mult_6x6) {
  // Create data
  size_t size = 6;
  size_t block = 3;

  std::vector<double> inputMatrOne{3,   -2, 5,  -19, -25, 25,  13, 22, -20, -24, -10, 11, -10, 9, -1, -13, 24,  -3,
                                   -12, 0,  -2, 1,   -19, -14, 16, 0,  24,  -12, -11, 18, 23,  2, 4,  -17, -16, -6};
  std::vector<double> inputMatrTwo{5,  -22, -20, 13, 5,   8,  -18, 3,   -3,  -22, -6,  -8, -12, -24, 17,  -25, 18, 20,
                                   18, 2,   10,  -4, -24, -5, 5,   -20, -22, -16, -13, -2, 13,  1,   -10, -20, 2,  -11};
  std::vector<double> outputMatr(size * size, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  // Add matrOne
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrOne.data()));
  taskDataSeq->inputs_count.emplace_back(inputMatrOne.size());

  // Add matrTwo
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrTwo.data()));
  taskDataSeq->inputs_count.emplace_back(inputMatrTwo.size());

  // Add size
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));

  // Add block
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block));

  // Add out matr
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputMatr.data()));
  taskDataSeq->outputs_count.emplace_back(outputMatr.size());

  auto resSeq = KuznetsovArtyomSeq::multMatrSquare(inputMatrOne, inputMatrTwo, size);

  // Create Task
  KuznetsovArtyomSeq::KuznetsovCannonMatrMultSeq testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  auto resSize = resSeq.size();

  for (size_t i = 0; i < resSize; ++i) {
    ASSERT_TRUE(KuznetsovArtyomSeq::isEqual(resSeq[i], outputMatr[i]));
  }
}

TEST(Kuznetsov_a_cannon_matr_mult_seq_func_tests, mult_7x7) {
  // Create data
  size_t size = 7;
  size_t block = 4;

  std::vector<double> inputMatrOne{21, -11, -17, -10, -3, -23, 3,   -6,  25, 18,  24,  4,  -24, -25, -12, 21,  -12,
                                   15, 14,  -23, 16,  -2, 21,  -17, -25, 18, -25, -3,  24, -14, -5,  -7,  -22, -23,
                                   -8, 25,  1,   5,   7,  -5,  3,   -15, 1,  -24, -22, 18, 18,  8,   -1};
  std::vector<double> inputMatrTwo{20, -25, 8,   -2,  -6, -11, 14, -21, -20, -16, -11, -11, 23,  8,  24, 20, -11,
                                   -7, -8,  -14, -18, 13, 21,  -6, -21, 15,  -21, -11, 8,   16,  15, 3,  22, 9,
                                   18, 18,  -4,  -24, 15, 20,  15, 9,   -2,  11,  15,  -13, -25, 13, 7};
  std::vector<double> outputMatr(size * size, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  // Add matrOne
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrOne.data()));
  taskDataSeq->inputs_count.emplace_back(inputMatrOne.size());

  // Add matrTwo
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrTwo.data()));
  taskDataSeq->inputs_count.emplace_back(inputMatrTwo.size());

  // Add size
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));

  // Add block
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block));

  // Add out matr
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputMatr.data()));
  taskDataSeq->outputs_count.emplace_back(outputMatr.size());

  auto resSeq = KuznetsovArtyomSeq::multMatrSquare(inputMatrOne, inputMatrTwo, size);

  // Create Task
  KuznetsovArtyomSeq::KuznetsovCannonMatrMultSeq testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  auto resSize = resSeq.size();

  for (size_t i = 0; i < resSize; ++i) {
    ASSERT_TRUE(KuznetsovArtyomSeq::isEqual(resSeq[i], outputMatr[i]));
  }
}
