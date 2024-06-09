// Copyright 2023 Safronov Mikhail
#include <gtest/gtest.h>

#include <vector>

#include "seq/safronov_m/include/ops_seq.hpp"

TEST(Safronov_mult_m_fox, validationTest) {
  size_t n = 2;
  std::vector<double> in1{2.0, 5.0, 3.0, -1.0};
  std::vector<double> in2{1.0, 0.0, 0.0, 1.0, 2.0};
  std::vector<double> out(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  SafronovSeqFoxAlgTask safronovTaskSequential(taskDataSeq);
  ASSERT_EQ(safronovTaskSequential.validation(), false);
}

TEST(Safronov_mult_m_fox, mult2) {
  size_t n = 10;
  std::vector<double> in1(n * n, 10.0);
  std::vector<double> in2(n * n, 1.0);
  std::vector<double> out(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  SafronovSeqFoxAlgTask safronovTaskSequential(taskDataSeq);
  ASSERT_EQ(safronovTaskSequential.validation(), true);
  safronovTaskSequential.pre_processing();
  safronovTaskSequential.run();
  safronovTaskSequential.post_processing();

  std::vector<double> expected_result = mulSafronov(in1, in2, n);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      EXPECT_DOUBLE_EQ(out[i * n + j], expected_result[i * n + j]);
    }
  }
}

TEST(Safronov_mult_m_fox, mult3) {
  size_t n = 10;
  std::vector<double> in1(n * n, 10.0);
  std::vector<double> in2(n * n, 10.0);
  std::vector<double> out(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  SafronovSeqFoxAlgTask safronovTaskSequential(taskDataSeq);
  ASSERT_EQ(safronovTaskSequential.validation(), true);
  safronovTaskSequential.pre_processing();
  safronovTaskSequential.run();
  safronovTaskSequential.post_processing();

  std::vector<double> expected_result = mulSafronov(in1, in2, n);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      EXPECT_DOUBLE_EQ(out[i * n + j], expected_result[i * n + j]);
    }
  }
}

TEST(Safronov_mult_m_fox, mult4) {
  size_t n = 3;
  std::vector<double> in1{1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0};
  std::vector<double> in2{2.0, 6.0, 18.0, 54.0, 162.0, 486.0, 1458.0, 4374.0, 13122.0};
  std::vector<double> out(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  SafronovSeqFoxAlgTask safronovTaskSequential(taskDataSeq);
  ASSERT_EQ(safronovTaskSequential.validation(), true);
  safronovTaskSequential.pre_processing();
  safronovTaskSequential.run();
  safronovTaskSequential.post_processing();

  std::vector<double> expected_result = mulSafronov(in1, in2, n);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      EXPECT_DOUBLE_EQ(out[i * n + j], expected_result[i * n + j]);
    }
  }
}

TEST(Safronov_mult_m_fox, mult5) {
  size_t n = 4;
  std::vector<double> in1{1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 3.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> in2{10.0, 9.0, 8.0, 7.0, 9.0, 8.0, 7.0, 6.0, 8.0, 7.0, 6.0, 5.0, 7.0, 6.0, 5.0, 4.0};
  std::vector<double> out(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in1.size());
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  SafronovSeqFoxAlgTask safronovTaskSequential(taskDataSeq);
  ASSERT_EQ(safronovTaskSequential.validation(), true);
  safronovTaskSequential.pre_processing();
  safronovTaskSequential.run();
  safronovTaskSequential.post_processing();

  std::vector<double> expected_result = mulSafronov(in1, in2, n);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      EXPECT_DOUBLE_EQ(out[i * n + j], expected_result[i * n + j]);
    }
  }
}