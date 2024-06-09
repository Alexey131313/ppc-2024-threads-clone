// Copyright 2024 Sokolova Daria
#include <gtest/gtest.h>

#include <vector>

#include "seq/sokolova_d_linear_image_filtering/include/ops_seq.hpp"

TEST(sokolova_d_linear_image_filtering, white_image) {
  uint32_t width = 32;
  uint32_t height = 32;

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
  GaussFilterHorizontal gaussFilterHorizontal(taskDataSeq);
  ASSERT_EQ(gaussFilterHorizontal.validation(), true);
  gaussFilterHorizontal.pre_processing();
  gaussFilterHorizontal.run();
  gaussFilterHorizontal.post_processing();
  ASSERT_EQ(expected, out);
}

TEST(sokolova_d_linear_image_filtering, black_image) {
  uint32_t width = 32;
  uint32_t height = 32;

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
  GaussFilterHorizontal gaussFilterHorizontal(taskDataSeq);
  ASSERT_EQ(gaussFilterHorizontal.validation(), true);
  gaussFilterHorizontal.pre_processing();
  gaussFilterHorizontal.run();
  gaussFilterHorizontal.post_processing();
  ASSERT_EQ(expected, out);
}

TEST(sokolova_d_linear_image_filtering, low_resolution_image) {
  uint32_t width = 2;
  uint32_t height = 2;

  // Create data
  std::vector<uint8_t> in(width * height * 3, 127);
  std::vector<uint8_t> out(width * height * 3, 0);
  std::vector<uint8_t> expected(width * height * 3, 127);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  GaussFilterHorizontal gaussFilterHorizontal(taskDataSeq);
  ASSERT_EQ(gaussFilterHorizontal.validation(), false);
}

TEST(sokolova_d_linear_image_filtering, high_resolution_image) {
  uint32_t width = 512;
  uint32_t height = 512;

  // Create data
  std::vector<uint8_t> in(width * height * 3, 127);
  std::vector<uint8_t> out(width * height * 3, 0);
  std::vector<uint8_t> expected(width * height * 3, 127);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  GaussFilterHorizontal gaussFilterHorizontal(taskDataSeq);
  ASSERT_EQ(gaussFilterHorizontal.validation(), true);
  gaussFilterHorizontal.pre_processing();
  gaussFilterHorizontal.run();
  gaussFilterHorizontal.post_processing();
  ASSERT_EQ(expected, out);
}

TEST(sokolova_d_linear_image_filtering, empty_image) {
  uint32_t width = 0;
  uint32_t height = 0;

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
  GaussFilterHorizontal gaussFilterHorizontal(taskDataSeq);
  ASSERT_EQ(gaussFilterHorizontal.validation(), false);
}