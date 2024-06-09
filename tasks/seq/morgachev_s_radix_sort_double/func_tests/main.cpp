// Copyright 2024 Morgachev Stepan
#include <gtest/gtest.h>

#include <vector>

#include "seq/morgachev_s_radix_sort_double/include/ops_seq.hpp"

TEST(morgachev_s_radix_sort_double_seq, test_one_number) {
  std::vector<double> in{1.0};
  std::vector<double> sorted{1.0};
  std::vector<double> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  RadixSortDoubleSequential test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], sorted[i]);
  }
}

TEST(morgachev_s_radix_sort_double_seq, test_with_simple_values) {
  std::vector<double> in{3.0, 2.0, 1.0};
  std::vector<double> sorted{1.0, 2.0, 3.0};
  std::vector<double> out(3);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  RadixSortDoubleSequential test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], sorted[i]);
  }
}

TEST(morgachev_s_radix_sort_double_seq, test_sorted_values) {
  std::vector<double> in{0e-8, 1.0, 2.0, 3.0, 10.0, 1e+8};
  std::vector<double> sorted{0e-8, 1.0, 2.0, 3.0, 10.0, 1e+8};
  std::vector<double> out(6);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  RadixSortDoubleSequential test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], sorted[i]);
  }
}

TEST(morgachev_s_radix_sort_double_seq, test_reversed_values) {
  std::vector<double> in{1e+8, 10.0, 3.0, 2.0, 1.0, 0e-8};
  std::vector<double> sorted{0e-8, 1.0, 2.0, 3.0, 10.0, 1e+8};
  std::vector<double> out(6);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  RadixSortDoubleSequential test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], sorted[i]);
  }
}

TEST(morgachev_s_radix_sort_double_seq, test_with_duplicate_values) {
  std::vector<double> in{3.5, 3.5, 17.03, 0.3, 12.4, 2.1, 0.3, 0.3};
  std::vector<double> sorted{0.3, 0.3, 0.3, 2.1, 3.5, 3.5, 12.4, 17.03};
  std::vector<double> out(8);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  RadixSortDoubleSequential test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], sorted[i]);
  }
}
