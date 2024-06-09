
// Copyright 2024 Konovalov Igor
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "omp/konovalov_i_radix_sort_doubles_s_m/include/ops_omp.hpp"

TEST(konovalov_i_double_radix_sort_seq, one_value_test) {
  // Create data
  std::vector<double> in = {0.0};
  std::vector<double> out = {0.0};

  std::vector<double *> res(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  RadixSortSequentialTask testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_EQ(out[0], res[0][0]);
}

TEST(konovalov_i_double_radix_sort_seq, sorted_order_test) {
  // Create data
  std::vector<double> in = {0.0, 1.0, 2.0, 2.2, 3.0};
  std::vector<double> out = {0.0, 1.0, 2.0, 2.2, 3.0};

  std::vector<double *> res(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  RadixSortSequentialTask testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < 5; i++) {
    if (out[i] != res[0][i]) {
      FAIL();
    }
  }
  SUCCEED();
}

TEST(konovalov_i_double_radix_sort_seq, reverse_order_test) {
  // Create data
  std::vector<double> in = {3.0, 2.2, 2.0, 1.0, 0.0};
  std::vector<double> out = {0.0, 1.0, 2.0, 2.2, 3.0};

  std::vector<double *> res(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  RadixSortSequentialTask testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < 5; i++) {
    if (out[i] != res[0][i]) {
      FAIL();
    }
  }
  SUCCEED();
}

TEST(konovalov_i_double_radix_sort_seq, random_order_test) {
  // Create data
  const size_t size = 5;
  std::mt19937 gen(1.0);
  std::uniform_real_distribution<> dis(0.0, 1e+5);
  std::vector<double> in(size);
  std::vector<double> out(size);
  for (size_t i = 0; i < size; i++) {
    double a = dis(gen);
    in[i] = a;
    out[i] = a;
  }
  std::sort(out.begin(), out.end());

  std::vector<double *> res(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  RadixSortSequentialTask testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < size; i++) {
    if (out[i] != res[0][i]) {
      FAIL();
    }
  }
  SUCCEED();
}

TEST(konovalov_i_double_radix_sort_seq, large_amount_random_test) {
  // Create data
  const size_t size = 1e+5;
  std::mt19937 gen(1.0);
  std::uniform_real_distribution<> dis(0.0, 1e+5);
  std::vector<double> in(size);
  std::vector<double> out(size);
  for (size_t i = 0; i < size; i++) {
    double a = dis(gen);
    in[i] = a;
    out[i] = a;
  }
  std::sort(out.begin(), out.end());

  std::vector<double *> res(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  RadixSortSequentialTask testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < size; i++) {
    if (out[i] != res[0][i]) {
      FAIL();
    }
  }
  SUCCEED();
}

// omp tests
TEST(konovalov_i_double_radix_sort_omp, one_value_test_parallel) {
  // Create data
  std::vector<double> in = {0.0};
  std::vector<double> out = {0.0};

  std::vector<double *> res(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  RadixSortOMPTaskParallel testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_EQ(out[0], res[0][0]);
}

TEST(konovalov_i_double_radix_sort_omp, sorted_order_test_parallel) {
  // Create data
  std::vector<double> in = {0.0, 1.0, 2.0, 2.2, 3.0};
  std::vector<double> out = {0.0, 1.0, 2.0, 2.2, 3.0};

  std::vector<double *> res(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  RadixSortSequentialTask testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < 5; i++) {
    if (out[i] != res[0][i]) {
      FAIL();
    }
  }
  SUCCEED();
}

TEST(konovalov_i_double_radix_sort_omp, reverse_order_test_parallel) {
  // Create data
  std::vector<double> in = {3.0, 2.2, 2.0, 1.0, 0.0};
  std::vector<double> out = {0.0, 1.0, 2.0, 2.2, 3.0};

  std::vector<double *> res(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  RadixSortOMPTaskParallel testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < 5; i++) {
    if (out[i] != res[0][i]) {
      FAIL();
    }
  }
  SUCCEED();
}

TEST(konovalov_i_double_radix_sort_omp, random_order_test_parallel) {
  // Create data
  const size_t size = 5;
  std::mt19937 gen(1.0);
  std::uniform_real_distribution<> dis(0.0, 1e+5);
  std::vector<double> in(size);
  std::vector<double> out(size);
  for (size_t i = 0; i < size; i++) {
    double a = dis(gen);
    in[i] = a;
    out[i] = a;
  }
  std::sort(out.begin(), out.end());

  std::vector<double *> res(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  RadixSortOMPTaskParallel testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < size; i++) {
    if (out[i] != res[0][i]) {
      FAIL();
    }
  }
  SUCCEED();
}

TEST(konovalov_i_double_radix_sort_omp, large_amount_random_test_parallel) {
  // Create data
  const size_t size = 1e+5;
  std::mt19937 gen(1.0);
  std::uniform_real_distribution<> dis(0.0, 1e+5);
  std::vector<double> in(size);
  std::vector<double> out(size);
  for (size_t i = 0; i < size; i++) {
    double a = dis(gen);
    in[i] = a;
    out[i] = a;
  }
  std::sort(out.begin(), out.end());

  std::vector<double *> res(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  RadixSortOMPTaskParallel testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < size; i++) {
    if (out[i] != res[0][i]) {
      FAIL();
    }
  }
  SUCCEED();
}