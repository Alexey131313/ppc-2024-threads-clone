// Copyright 2024 Mukhin Ivan
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "omp/mukhin_a_gaussian_filter/include/gaussian_filter.hpp"
#include "omp/mukhin_a_gaussian_filter/include/pixel_map.hpp"

TEST(mukhin_i_a_gaussian_filter_block, can_work_with_black_pixel_map) {
  // Create data
  uint32_t width = 32;
  uint32_t height = 32;

  PixelMap in(width, height);
  PixelMap out(width, height);
  PixelMap expected(width, height);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  mukhin_i_omp::GaussianFilterOMP testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, expected);
}

TEST(mukhin_i_a_gaussian_filter_block, can_work_with_white_pixel_map) {
  // Create data
  uint32_t width = 32;
  uint32_t height = 32;
  PixelMap in(width, height, 255, 255, 255);
  PixelMap out(width, height, 255, 255, 255);
  PixelMap expected(width, height, 255, 255, 255);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  mukhin_i_omp::GaussianFilterOMP testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  ASSERT_EQ(testTaskSequential.run(), true);
  testTaskSequential.post_processing();
}

TEST(mukhin_i_a_gaussian_filter_block, can_work_with_middle_pixel_map) {
  // Create data
  uint32_t width = 32;
  uint32_t height = 32;
  PixelMap in(width, height, 128, 128, 128);
  PixelMap out(width, height, 128, 128, 128);
  PixelMap expected(width, height, 128, 128, 128);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  mukhin_i_omp::GaussianFilterOMP testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  ASSERT_EQ(testTaskSequential.run(), true);
  testTaskSequential.post_processing();
}

TEST(mukhin_i_a_gaussian_filter_block, can_work_with_small_pixel_map) {
  // Create data
  uint32_t width = 6;
  uint32_t height = 6;
  PixelMap in(width, height);
  PixelMap out(width, height);
  PixelMap expected(width, height);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  mukhin_i_omp::GaussianFilterOMP testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, expected);
}

TEST(mukhin_i_a_gaussian_filter_block, can_work_with_big_pixel_map) {
  // Create data
  uint32_t width = 300;
  uint32_t height = 300;
  PixelMap in(width, height);
  PixelMap out(width, height);
  PixelMap expected(width, height);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  mukhin_i_omp::GaussianFilterOMP testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, expected);
}
