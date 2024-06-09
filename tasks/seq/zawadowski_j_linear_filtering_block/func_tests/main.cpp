// Copyright 2024 Zawadowski Jan

#include "seq/zawadowski_j_linear_filtering_block/include/linear_filtering_block.hpp"

TEST(zawadowski_j_linear_filtering_block, test_empty_image) {
  // Create data
  uint16_t width = 6;
  uint16_t height = 6;
  std::vector<uint8_t> image(width * height);
  std::vector<uint8_t> result(width * height);
  std::vector<uint8_t> sample(width * height);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  LinearFiltering testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(result, sample);
}

TEST(zawadowski_j_linear_filtering_block, test_size_smaller_than_kernel) {
  // Create data
  uint16_t width = 2;
  uint16_t height = 2;
  std::vector<uint8_t> image(width * height);
  std::vector<uint8_t> result(width * height);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  LinearFiltering testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
  testTaskSequential.pre_processing();
  ASSERT_ANY_THROW(testTaskSequential.run());
  testTaskSequential.post_processing();
}

TEST(zawadowski_j_linear_filtering_block, test_size_smaller_than_block) {
  // Create data
  uint16_t width = 3;
  uint16_t height = 3;
  std::vector<uint8_t> image(width * height);
  std::vector<uint8_t> result(width * height);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  LinearFiltering testTaskSequential(taskDataSeq);
  testTaskSequential.blockSet(4, 4);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  ASSERT_ANY_THROW(testTaskSequential.run());
  testTaskSequential.post_processing();
}

TEST(zawadowski_j_linear_filtering_block, test_large_kernel) {
  // Create data
  uint16_t width = 3;
  uint16_t height = 3;
  std::vector<uint8_t> image(width * height);
  std::vector<uint8_t> result(width * height);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  LinearFiltering testTaskSequential(taskDataSeq);
  ASSERT_ANY_THROW(testTaskSequential.kernelSet({1, 2, 2, 1, 2, 4, 4, 2, 2, 4, 4, 2, 1, 2, 2, 1}));
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
}

TEST(zawadowski_j_linear_filtering_block, test_small_kernel) {
  // Create data
  uint16_t width = 3;
  uint16_t height = 3;
  std::vector<uint8_t> image(width * height);
  std::vector<uint8_t> result(width * height);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  // Create Task
  LinearFiltering testTaskSequential(taskDataSeq);
  ASSERT_ANY_THROW(testTaskSequential.kernelSet({1, 2, 2, 1}));
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
}
