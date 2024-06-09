// Copyright 2024 Ivanchenko Aleksei
#include <gtest/gtest.h>

#include <vector>

#include "seq/ivanchenko_a_gauss_filter_vertical/include/ops_seq.hpp"

TEST(ivanchenko_a_gauss_filter_vertical, black_image) {
  uint32_t width = 16;
  uint32_t height = 16;

  // Create data
  std::vector<uint8_t> in(width * height * 3, 0);
  std::vector<uint8_t> out(width * height * 3, 0);
  std::vector<uint8_t> expected(width * height * 3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  GaussFilterSequential gaussFilterSequential(taskDataSeq);
  ASSERT_EQ(gaussFilterSequential.validation(), true);
  gaussFilterSequential.pre_processing();
  gaussFilterSequential.run();
  gaussFilterSequential.post_processing();
  ASSERT_EQ(expected, out);
}
TEST(ivanchenko_a_gauss_filter_vertical, while_image) {
  uint32_t width = 16;
  uint32_t height = 16;

  // Create data
  std::vector<uint8_t> in(width * height * 3, 255);
  std::vector<uint8_t> out(width * height * 3, 0);
  std::vector<uint8_t> expected(width * height * 3, 255);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  GaussFilterSequential gaussFilterSequential(taskDataSeq);
  ASSERT_EQ(gaussFilterSequential.validation(), true);
  gaussFilterSequential.pre_processing();
  gaussFilterSequential.run();
  gaussFilterSequential.post_processing();
  ASSERT_EQ(expected, out);
}
TEST(ivanchenko_a_gauss_filter_vertical, monotone_image) {
  uint32_t width = 16;
  uint32_t height = 16;

  // Create data
  std::vector<uint8_t> in(width * height * 3);
  std::vector<uint8_t> out(width * height * 3, 0);
  std::vector<uint8_t> expected(width * height * 3);
  for (size_t i = 0; i < width * height * 3; i++) {
    in[i] = 64 + 64 * (i % 3);
    expected[i] = in[i];
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  GaussFilterSequential gaussFilterSequential(taskDataSeq);
  ASSERT_EQ(gaussFilterSequential.validation(), true);
  gaussFilterSequential.pre_processing();
  gaussFilterSequential.run();
  gaussFilterSequential.post_processing();
  ASSERT_EQ(expected, out);
}
TEST(ivanchenko_a_gauss_filter_vertical, chessboard_image) {
  uint32_t width = 16;
  uint32_t height = 16;

  // Create data
  std::vector<uint8_t> in(width * height * 3);
  std::vector<uint8_t> out(width * height * 3, 0);
  std::vector<uint8_t> expected(width * height * 3);
  for (size_t i = 0; i < width * height * 3; i += 3) {
    int x = i % width;
    int y = i / width;
    in[i] = in[i + 1] = in[i + 2] = 128 - ((x + y) % 2);
    expected[i] = expected[i + 1] = expected[i + 2] = 128;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  GaussFilterSequential gaussFilterSequential(taskDataSeq);
  ASSERT_EQ(gaussFilterSequential.validation(), true);
  gaussFilterSequential.pre_processing();
  gaussFilterSequential.run();
  gaussFilterSequential.post_processing();
  ASSERT_EQ(expected, out);
}
TEST(ivanchenko_a_gauss_filter_vertical, image_with_some_noise) {
  uint32_t width = 16;
  uint32_t height = 16;

  // Create data
  std::vector<uint8_t> in(width * height * 3);
  std::vector<uint8_t> out(width * height * 3, 0);
  std::vector<uint8_t> expected(width * height * 3);
  for (size_t i = 0; i < width * height * 3; i += 3) {
    int x = i % width;
    int y = i / width;
    in[i] = in[i + 1] = in[i + 2] = 128 - 3 * static_cast<int>((x + y) % 13 == 0);
    expected[i] = expected[i + 1] = expected[i + 2] = 128;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  GaussFilterSequential gaussFilterSequential(taskDataSeq);
  ASSERT_EQ(gaussFilterSequential.validation(), true);
  gaussFilterSequential.pre_processing();
  gaussFilterSequential.run();
  gaussFilterSequential.post_processing();
  ASSERT_EQ(expected, out);
}
