// Copyright 2023 Ulyanov Daniil
#include <gtest/gtest.h>

#include <vector>

#include "seq/ulyanov_d_var_24/include/ops_seq.hpp"

TEST(Sequential, test_ulyanov_1) {
  int height = 5;
  int width = 5;

  // Create data
  std::vector<uint8_t> in = {
      0, 0,   0,   0,   0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 128, 128, 128, 0,
      0, 0,   0,   0,   0, 0, 0, 0, 128, 128, 128, 255, 255, 255, 128, 128, 128, 0, 0, 0, 0, 0,   0,   0,   0,
      0, 128, 128, 128, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 0,   0,   0,   0,
  };
  std::vector<uint8_t> out(height * width * 3, 0);
  std::vector<uint8_t> res = {
      0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 55, 55, 55, 72, 72, 72, 55,
      55, 55, 0,  0,  0,  0,  0,  0, 72, 72, 72, 92, 92, 92, 72, 72, 72, 0, 0,  0,  0,  0,  0,  55, 55,
      55, 72, 72, 72, 55, 55, 55, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,
  };

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(in.data());
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(out.data());
  taskDataSeq->outputs_count.emplace_back(height);
  taskDataSeq->outputs_count.emplace_back(width);

  // Create Task
  FilterGaussHorizontalSequentialUlyanov TaskSequential(taskDataSeq);
  ASSERT_EQ(TaskSequential.validation(), true);
  TaskSequential.pre_processing();
  TaskSequential.run();
  TaskSequential.post_processing();

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], res[i]);
  }
}

TEST(Sequential, test_ulyanov_2) {
  int height = 5;
  int width = 5;

  // Create data
  std::vector<uint8_t> in = {
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 0,
      0,   0,   0,   0,   0,   0,   0,   0,   128, 128, 128, 128, 128, 128, 0,   0,   0,   0,   0,
      0,   0,   0,   0,   128, 128, 128, 128, 128, 128, 0,   0,   0,   0,   0,   0,   0,   0,   0,
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  };
  std::vector<uint8_t> out(height * width * 3, 0);
  std::vector<uint8_t> res = {
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 68,
      68,  68,  40,  40,  40,  68,  68,  68,  128, 128, 128, 128, 128, 128, 40,  40,  40,  0,   0,
      0,   40,  40,  40,  128, 128, 128, 128, 128, 128, 68,  68,  68,  40,  40,  40,  68,  68,  68,
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  };

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(height);
  taskDataSeq->outputs_count.emplace_back(width);

  // Create Task
  FilterGaussHorizontalSequentialUlyanov TaskSequential(taskDataSeq);
  ASSERT_EQ(TaskSequential.validation(), true);
  TaskSequential.pre_processing();
  TaskSequential.run();
  TaskSequential.post_processing();

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], res[i]);
  }
}

TEST(Sequential, test_ulyanov_3) {
  int height = 5;
  int width = 5;

  // Create data
  std::vector<uint8_t> in = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  std::vector<uint8_t> out(height * width * 3, 0);
  std::vector<uint8_t> res = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(height);
  taskDataSeq->outputs_count.emplace_back(width);

  // Create Task
  FilterGaussHorizontalSequentialUlyanov TaskSequential(taskDataSeq);
  ASSERT_EQ(TaskSequential.validation(), true);
  TaskSequential.pre_processing();
  TaskSequential.run();
  TaskSequential.post_processing();

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], res[i]);
  }
}

TEST(Sequential, test_ulyanov_4) {
  int height = 5;
  int width = 5;

  // Create data
  std::vector<uint8_t> in = {
      0,   0,   0, 0, 0,   0,   0,   0, 0, 0, 0, 0,   0,   0,   0, 0, 0, 0, 255, 255, 255, 0, 0, 0,   255,
      255, 255, 0, 0, 0,   0,   0,   0, 0, 0, 0, 128, 128, 128, 0, 0, 0, 0, 0,   0,   0,   0, 0, 255, 255,
      255, 0,   0, 0, 255, 255, 255, 0, 0, 0, 0, 0,   0,   0,   0, 0, 0, 0, 0,   0,   0,   0, 0, 0,   0,
  };
  std::vector<uint8_t> out(height * width * 3, 0);
  std::vector<uint8_t> res = {
      0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,   0,   0,   0,  0,  0,  0, 46, 46, 46, 73, 73, 73, 46,
      46, 46, 0,  0,  0,  0,  0,  0, 73, 73, 73, 120, 120, 120, 73, 73, 73, 0, 0,  0,  0,  0,  0,  46, 46,
      46, 73, 73, 73, 46, 46, 46, 0, 0,  0,  0,  0,   0,   0,   0,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,
  };

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(height);
  taskDataSeq->outputs_count.emplace_back(width);

  // Create Task
  FilterGaussHorizontalSequentialUlyanov TaskSequential(taskDataSeq);
  ASSERT_EQ(TaskSequential.validation(), true);
  TaskSequential.pre_processing();
  TaskSequential.run();
  TaskSequential.post_processing();

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], res[i]);
  }
}

TEST(Sequential, test_ulyanov_5) {
  int height = 5;
  int width = 5;

  // Create data
  std::vector<uint8_t> in = {
      0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   128,
      128, 128, 128, 128, 128, 128, 128, 128, 0, 0, 0,   0,   0,   0,   128, 128, 128, 128, 128,
      128, 128, 128, 128, 0,   0,   0,   0,   0, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128,
      0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0,   0,   0,   0,   0,   0,   0,   0,
  };
  std::vector<uint8_t> out(height * width * 3, 0);
  std::vector<uint8_t> res = {
      0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,   0,   0,   0,  0,  0,  0, 59, 59, 59, 87, 87, 87, 59,
      59, 59, 0,  0,  0,  0,  0,  0, 87, 87, 87, 127, 127, 127, 87, 87, 87, 0, 0,  0,  0,  0,  0,  59, 59,
      59, 87, 87, 87, 59, 59, 59, 0, 0,  0,  0,  0,   0,   0,   0,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,
  };

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(height);
  taskDataSeq->outputs_count.emplace_back(width);

  // Create Task
  FilterGaussHorizontalSequentialUlyanov TaskSequential(taskDataSeq);
  ASSERT_EQ(TaskSequential.validation(), true);
  TaskSequential.pre_processing();
  TaskSequential.run();
  TaskSequential.post_processing();

  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(out[i], res[i]);
  }
}

// int main(int argc, char **argv) {
//   testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
