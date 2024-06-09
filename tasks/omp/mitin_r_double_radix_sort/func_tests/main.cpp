// Copyright 2024 Mitin Roman
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "omp/mitin_r_double_radix_sort/include/ops_omp.hpp"

TEST(mitin_r_double_radix_sort_omp, Test_Sort_two_value_test) {
  // Create data
  std::vector<double> in{2.0, 1.0};
  std::vector<double> expected{1.0, 2.0};

  std::vector<double *> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  SortRadixDoubleTaskOMP test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[0][i], expected[i]);
  }
}

TEST(mitin_r_double_radix_sort_omp, Test_Sort_simple) {
  // Create data
  std::vector<double> in{0.5, 5.0, 1.0, 0.3, 10.0};
  std::vector<double> expected{0.3, 0.5, 1.0, 5.0, 10.0};

  std::vector<double *> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  SortRadixDoubleTaskOMP test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[0][i], expected[i]);
  }
}

TEST(mitin_r_double_radix_sort_omp, Test_Sort_sorted) {
  // Create data
  std::vector<double> in{0, 0e-10, 0.3, 0.5, 1.0, 5.0, 10.0, 100.0, 1e+8};
  std::vector<double> expected{0, 0e-10, 0.3, 0.5, 1.0, 5.0, 10.0, 100.0, 1e+8};

  std::vector<double *> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  SortRadixDoubleTaskOMP test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[0][i], expected[i]);
  }
}

TEST(mitin_r_double_radix_sort_omp, Test_Sort_reverse_order) {
  // Create data
  std::vector<double> in{10.0, 8.0, 4.0, 2.0, 1.0, 0.0};
  std::vector<double> expected{0.0, 1.0, 2.0, 4.0, 8.0, 10.0};

  std::vector<double *> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  SortRadixDoubleTaskOMP test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[0][i], expected[i]);
  }
}

TEST(mitin_r_double_radix_sort_omp, Test_Sort_random_test) {
  // Create data
  constexpr size_t input_size = 1e+5;
  std::mt19937 gen(1.0);
  std::uniform_real_distribution<> dis(0.0, 1e+10);
  std::vector<double> in(input_size);
  std::vector<double> expected(input_size);
  for (size_t i = 0; i < input_size; i++) {
    in[i] = expected[i] = dis(gen);
  }

  std::vector<double *> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  SortRadixDoubleTaskOMP test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  std::sort(expected.begin(), expected.end());

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[0][i], expected[i]);
  }
}
