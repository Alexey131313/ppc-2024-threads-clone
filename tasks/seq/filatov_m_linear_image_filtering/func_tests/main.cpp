// Copyright 2024 Filatov Maxim
#include <gtest/gtest.h>

#include <vector>

#include "seq/filatov_m_linear_image_filtering/include/ops_seq.hpp"

TEST(filatov_m_linear_image_filtering, white_image) {
  // Define dimensions
  const uint64_t width = 32;
  const uint64_t height = 32;

  // Initialize data vectors
  std::vector<uint8_t> inputData(width * height * 3, 255);
  std::vector<uint8_t> outputData(width * height * 3, 0);
  std::vector<uint8_t> expectedData(width * height * 3, 255);

  // Prepare TaskData
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskData->inputs_count.push_back(width);
  taskData->inputs_count.push_back(height);
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(outputData.data()));
  taskData->outputs_count.push_back(width);
  taskData->outputs_count.push_back(height);

  // Instantiate and run GaussFilterHorizontal
  GaussFilterHorizontal gaussFilter(taskData);
  ASSERT_TRUE(gaussFilter.validation());
  gaussFilter.pre_processing();
  gaussFilter.run();
  gaussFilter.post_processing();

  // Check the result
  ASSERT_EQ(expectedData, outputData);
}

TEST(filatov_m_linear_image_filtering, black_image) {
  uint64_t width = 32;
  uint64_t height = 32;

  // Create data
  std::vector<uint8_t> in(width * height * 3, 0);
  std::vector<uint8_t> out(width * height * 3, 0);
  std::vector<uint8_t> expected(width * height * 3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskImageData = std::make_shared<ppc::core::TaskData>();
  taskImageData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskImageData->inputs_count.emplace_back(width);
  taskImageData->inputs_count.emplace_back(height);
  taskImageData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskImageData->outputs_count.emplace_back(width);
  taskImageData->outputs_count.emplace_back(height);

  // Create Task
  GaussFilterHorizontal gaussFilterHorizontal(taskImageData);
  ASSERT_EQ(gaussFilterHorizontal.validation(), true);
  gaussFilterHorizontal.pre_processing();
  gaussFilterHorizontal.run();
  gaussFilterHorizontal.post_processing();
  ASSERT_EQ(expected, out);
}

TEST(filatov_m_linear_image_filtering, low_resolution_image) {
  uint64_t width = 2;
  uint64_t height = 2;

  // Create data
  std::vector<uint8_t> in(width * height * 3, 127);
  std::vector<uint8_t> out(width * height * 3, 0);
  std::vector<uint8_t> expected(width * height * 3, 127);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskImageData = std::make_shared<ppc::core::TaskData>();
  taskImageData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskImageData->inputs_count.emplace_back(width);
  taskImageData->inputs_count.emplace_back(height);
  taskImageData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskImageData->outputs_count.emplace_back(width);
  taskImageData->outputs_count.emplace_back(height);

  // Create Task
  GaussFilterHorizontal gaussFilterHorizontal(taskImageData);
  ASSERT_EQ(gaussFilterHorizontal.validation(), false);
}

TEST(filatov_m_linear_image_filtering, big_image) {
  uint64_t width = 512;
  uint64_t height = 512;

  // Create data
  std::vector<uint8_t> in(width * height * 3, 127);
  std::vector<uint8_t> out(width * height * 3, 0);
  std::vector<uint8_t> expected(width * height * 3, 127);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskImageData = std::make_shared<ppc::core::TaskData>();
  taskImageData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskImageData->inputs_count.emplace_back(width);
  taskImageData->inputs_count.emplace_back(height);
  taskImageData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskImageData->outputs_count.emplace_back(width);
  taskImageData->outputs_count.emplace_back(height);

  // Create Task
  GaussFilterHorizontal gaussFilterHorizontal(taskImageData);
  ASSERT_EQ(gaussFilterHorizontal.validation(), true);
  gaussFilterHorizontal.pre_processing();
  gaussFilterHorizontal.run();
  gaussFilterHorizontal.post_processing();
  ASSERT_EQ(expected, out);
}

TEST(filatov_m_linear_image_filtering, red_image) {
  uint64_t width = 32;
  uint64_t height = 32;

  // Create data
  std::vector<uint8_t> in(width * height * 3);
  std::fill(in.begin(), in.end(), 0);
  for (size_t i = 0; i < in.size(); i += 3) {
    in[i] = 255;  // Red channel
  }
  std::vector<uint8_t> out(width * height * 3, 0);
  std::vector<uint8_t> expected = in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskImageData = std::make_shared<ppc::core::TaskData>();
  taskImageData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskImageData->inputs_count.emplace_back(width);
  taskImageData->inputs_count.emplace_back(height);
  taskImageData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskImageData->outputs_count.emplace_back(width);
  taskImageData->outputs_count.emplace_back(height);

  // Create Task
  GaussFilterHorizontal gaussFilterHorizontal(taskImageData);
  ASSERT_EQ(gaussFilterHorizontal.validation(), true);
  gaussFilterHorizontal.pre_processing();
  gaussFilterHorizontal.run();
  gaussFilterHorizontal.post_processing();
  ASSERT_EQ(expected, out);
}

TEST(filatov_m_linear_image_filtering, green_image) {
  uint64_t width = 32;
  uint64_t height = 32;

  // Create data
  std::vector<uint8_t> in(width * height * 3);
  std::fill(in.begin(), in.end(), 0);
  for (size_t i = 1; i < in.size(); i += 3) {
    in[i] = 255;  // Red channel
  }
  std::vector<uint8_t> out(width * height * 3, 0);
  std::vector<uint8_t> expected = in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskImageData = std::make_shared<ppc::core::TaskData>();
  taskImageData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskImageData->inputs_count.emplace_back(width);
  taskImageData->inputs_count.emplace_back(height);
  taskImageData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskImageData->outputs_count.emplace_back(width);
  taskImageData->outputs_count.emplace_back(height);

  // Create Task
  GaussFilterHorizontal gaussFilterHorizontal(taskImageData);
  ASSERT_EQ(gaussFilterHorizontal.validation(), true);
  gaussFilterHorizontal.pre_processing();
  gaussFilterHorizontal.run();
  gaussFilterHorizontal.post_processing();
  ASSERT_EQ(expected, out);
}

TEST(filatov_m_linear_image_filtering, blue_image) {
  uint64_t width = 32;
  uint64_t height = 32;

  // Create data
  std::vector<uint8_t> in(width * height * 3);
  std::fill(in.begin(), in.end(), 0);
  for (size_t i = 2; i < in.size(); i += 3) {
    in[i] = 255;  // Red channel
  }
  std::vector<uint8_t> out(width * height * 3, 0);
  std::vector<uint8_t> expected = in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskImageData = std::make_shared<ppc::core::TaskData>();
  taskImageData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskImageData->inputs_count.emplace_back(width);
  taskImageData->inputs_count.emplace_back(height);
  taskImageData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskImageData->outputs_count.emplace_back(width);
  taskImageData->outputs_count.emplace_back(height);

  // Create Task
  GaussFilterHorizontal gaussFilterHorizontal(taskImageData);
  ASSERT_EQ(gaussFilterHorizontal.validation(), true);
  gaussFilterHorizontal.pre_processing();
  gaussFilterHorizontal.run();
  gaussFilterHorizontal.post_processing();
  ASSERT_EQ(expected, out);
}

TEST(filatov_m_linear_image_filtering, invalid_dimensions) {
  uint64_t width = 0;
  uint64_t height = 0;

  // Create data
  std::vector<uint8_t> in(width * height * 3, 0);
  std::vector<uint8_t> out(width * height * 3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskImageData = std::make_shared<ppc::core::TaskData>();
  taskImageData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskImageData->inputs_count.emplace_back(width);
  taskImageData->inputs_count.emplace_back(height);
  taskImageData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskImageData->outputs_count.emplace_back(width);
  taskImageData->outputs_count.emplace_back(height);

  // Create Task
  GaussFilterHorizontal gaussFilterHorizontal(taskImageData);
  ASSERT_EQ(gaussFilterHorizontal.validation(), false);
}
