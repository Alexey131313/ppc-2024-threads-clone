// Copyright 2024 Pivovarov Alexey
#include <gtest/gtest.h>

#include <vector>

#include "stl/pivovarov_a_strassen_alg/include/ops_stl.hpp"

TEST(Pivovarov_a_strassen_alg_stl, Multiplication_4x4) {
  int n = 4;

  // Create data
  std::vector<double> in_A = createRndMatrix(n);
  std::vector<double> in_B = createRndMatrix(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = multiplyMatrix(in_A, in_B, n);

  // Create Task
  TestTaskSTLParallelPivovarovStrassen test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(std::round(res[i] * std::pow(10, 3)) / std::pow(10, 3),
              std::round(out[i] * std::pow(10, 3)) / std::pow(10, 3));
  }
}

TEST(Pivovarov_a_strassen_alg_stl, Multiplication_8x8) {
  int n = 8;

  // Create data
  std::vector<double> in_A = createRndMatrix(n);
  std::vector<double> in_B = createRndMatrix(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = multiplyMatrix(in_A, in_B, n);

  // Create Task
  TestTaskSTLParallelPivovarovStrassen test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(std::round(res[i] * std::pow(10, 3)) / std::pow(10, 3),
              std::round(out[i] * std::pow(10, 3)) / std::pow(10, 3));
  }
}

TEST(Pivovarov_a_strassen_alg_stl, Multiplication_16x16) {
  int n = 16;

  // Create data
  std::vector<double> in_A = createRndMatrix(n);
  std::vector<double> in_B = createRndMatrix(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = multiplyMatrix(in_A, in_B, n);

  // Create Task
  TestTaskSTLParallelPivovarovStrassen test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(std::round(res[i] * std::pow(10, 3)) / std::pow(10, 3),
              std::round(out[i] * std::pow(10, 3)) / std::pow(10, 3));
  }
}

TEST(Pivovarov_a_strassen_alg_stl, Multiplication_32x32) {
  int n = 32;

  // Create data
  std::vector<double> in_A = createRndMatrix(n);
  std::vector<double> in_B = createRndMatrix(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = multiplyMatrix(in_A, in_B, n);

  // Create Task
  TestTaskSTLParallelPivovarovStrassen test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(std::round(res[i] * std::pow(10, 3)) / std::pow(10, 3),
              std::round(out[i] * std::pow(10, 3)) / std::pow(10, 3));
  }
}

TEST(Pivovarov_a_strassen_alg_stl, Multiplication_64x64) {
  int n = 64;

  // Create data
  std::vector<double> in_A = createRndMatrix(n);
  std::vector<double> in_B = createRndMatrix(n);
  std::vector<double> out(n * n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(in_A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(in_B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(&n)));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> res = multiplyMatrix(in_A, in_B, n);

  // Create Task
  TestTaskSTLParallelPivovarovStrassen test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(std::round(res[i] * std::pow(10, 3)) / std::pow(10, 3),
              std::round(out[i] * std::pow(10, 3)) / std::pow(10, 3));
  }
}
