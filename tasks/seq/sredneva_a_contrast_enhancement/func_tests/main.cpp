// Copyright 2023 Sredneva Anastasiya
#include <gtest/gtest.h>

#include <vector>

#include "seq/sredneva_a_contrast_enhancement/include/ops_seq.hpp"

TEST(sredneva_a_contrast_enhancement_seq, test_1) {
  int n = 3;
  int m = 3;

  // Create data
  std::vector<uint8_t> in = {200, 67, 120, 88, 90, 199, 65, 98, 77};

  uint8_t min = *std::min_element(in.begin(), in.end());
  uint8_t max = *std::max_element(in.begin(), in.end());

  std::vector<int> in2 = {n, m};
  std::vector<uint8_t> in3 = {min, max};
  std::vector<uint8_t> out(n * m);

  std::vector<uint8_t> res = {255, 3, 103, 43, 47, 253, 0, 62, 22};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataSeq->inputs_count.emplace_back(in3.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ContrastEnhancementSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  ASSERT_EQ(testTaskSequential.run(), true);
  testTaskSequential.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(res[i], out[i]);
  }
}

TEST(sredneva_a_contrast_enhancement_seq, test_2) {
  int n = 4;
  int m = 3;

  // Create data
  std::vector<uint8_t> in = {67, 120, 88, 90, 199, 65, 98, 77, 0, 16, 99, 255};

  uint8_t min = *std::min_element(in.begin(), in.end());
  uint8_t max = *std::max_element(in.begin(), in.end());

  std::vector<int> in2 = {n, m};
  std::vector<uint8_t> in3 = {min, max};
  std::vector<uint8_t> out(n * m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataSeq->inputs_count.emplace_back(in3.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ContrastEnhancementSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  ASSERT_EQ(testTaskSequential.run(), true);
  testTaskSequential.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(in[i], out[i]);
  }
}

TEST(sredneva_a_contrast_enhancement_seq, test_3_rand) {
  int n = 5;
  int m = 4;
  uint8_t min = 75;
  uint8_t max = 200;

  // Create data
  std::vector<uint8_t> in = getRandomPicture(n, m, min, max);

  std::vector<int> in2 = {n, m};
  std::vector<uint8_t> in3 = {min, max};
  std::vector<uint8_t> out(n * m);

  std::vector<uint8_t> input(in);
  std::vector<uint8_t> res(n * m, 0);
  for (int i = 0; i < n * m; i++) {
    res[i] = (input[i] - min) * 255 / (max - min);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataSeq->inputs_count.emplace_back(in3.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ContrastEnhancementSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  ASSERT_EQ(testTaskSequential.run(), true);
  testTaskSequential.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(res[i], out[i]);
  }
}

TEST(sredneva_a_contrast_enhancement_seq, test_4_all_5) {
  int n = 3;
  int m = 2;

  // Create data
  std::vector<uint8_t> in(n * m, 5);

  uint8_t min = *std::min_element(in.begin(), in.end());
  uint8_t max = *std::max_element(in.begin(), in.end());

  std::vector<int> in2 = {n, m};
  std::vector<uint8_t> in3 = {min, max};
  std::vector<uint8_t> out(n * m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataSeq->inputs_count.emplace_back(in3.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ContrastEnhancementSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  ASSERT_EQ(testTaskSequential.run(), false);
  testTaskSequential.post_processing();
}

TEST(sredneva_a_contrast_enhancement_seq, test_5) {
  int n = 2;
  int m = 4;

  // Create data
  std::vector<uint8_t> in = {254, 1, 254, 1, 254, 1, 254, 1};

  uint8_t min = *std::min_element(in.begin(), in.end());
  uint8_t max = *std::max_element(in.begin(), in.end());

  std::vector<int> in2 = {n, m};
  std::vector<uint8_t> in3 = {min, max};
  std::vector<uint8_t> out(n * m);

  std::vector<uint8_t> res = {255, 0, 255, 0, 255, 0, 255, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataSeq->inputs_count.emplace_back(in3.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ContrastEnhancementSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  ASSERT_EQ(testTaskSequential.run(), true);
  testTaskSequential.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(res[i], out[i]);
  }
}
