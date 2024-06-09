// Copyright 2024 Lapin Dmitriy
#include <gtest/gtest.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>

#include "tbb/lapin_d_image_filter_block/include/ops_tbb.hpp"

TEST(lapin_d_image_filter_block_parallel, Test_Lapin_test_6x8) {
  // Create data
  int height = 6;
  int width = 8;
  std::vector<std::vector<int>> in(height);
  for (int i = 0; i < height; i++) {
    in[i].resize(width);
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      in[i][j] = rand() % 256;
    }
  }

  std::vector<std::vector<int>> out(height);
  for (int i = 0; i < height; i++) {
    out[i].resize(width);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));

  // Create Task
  BlockFilterTBBTaskSequential blockFilterTBBTaskSequential(taskDataSeq);
  ASSERT_EQ(blockFilterTBBTaskSequential.validation(), true);
  blockFilterTBBTaskSequential.pre_processing();
  blockFilterTBBTaskSequential.run();
  blockFilterTBBTaskSequential.post_processing();

  std::vector<std::vector<int>> out_2(height);
  for (int i = 0; i < height; i++) {
    out_2[i].resize(width);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataPar->inputs_count.emplace_back(height);
  taskDataPar->inputs_count.emplace_back(width);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_2));

  // Create Task
  BlockFilterTBBTaskParallel blockFilterTBBTaskParallel(taskDataPar);
  ASSERT_EQ(blockFilterTBBTaskParallel.validation(), true);
  blockFilterTBBTaskParallel.pre_processing();
  blockFilterTBBTaskParallel.run();
  blockFilterTBBTaskParallel.post_processing();
  ASSERT_EQ(out[0][0], out_2[0][0]);
}

TEST(lapin_d_image_filter_block_parallel, Test_Lapin_test_60x80) {
  // Create data
  int height = 60;
  int width = 80;
  std::vector<std::vector<int>> in(height);
  for (int i = 0; i < height; i++) {
    in[i].resize(width);
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      in[i][j] = rand() % 256;
    }
  }

  std::vector<std::vector<int>> out(height);
  for (int i = 0; i < height; i++) {
    out[i].resize(width);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));

  // Create Task
  BlockFilterTBBTaskSequential blockFilterTBBTaskSequential(taskDataSeq);
  ASSERT_EQ(blockFilterTBBTaskSequential.validation(), true);
  blockFilterTBBTaskSequential.pre_processing();
  blockFilterTBBTaskSequential.run();
  blockFilterTBBTaskSequential.post_processing();

  std::vector<std::vector<int>> out_2(height);
  for (int i = 0; i < height; i++) {
    out_2[i].resize(width);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataPar->inputs_count.emplace_back(height);
  taskDataPar->inputs_count.emplace_back(width);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_2));

  // Create Task
  BlockFilterTBBTaskParallel blockFilterTBBTaskParallel(taskDataPar);
  ASSERT_EQ(blockFilterTBBTaskParallel.validation(), true);
  blockFilterTBBTaskParallel.pre_processing();
  blockFilterTBBTaskParallel.run();
  blockFilterTBBTaskParallel.post_processing();
  ASSERT_EQ(out[0][0], out_2[0][0]);
}

TEST(lapin_d_image_filter_block_parallel, Test_Lapin_test_100x200) {
  // Create data
  int height = 100;
  int width = 200;
  std::vector<std::vector<int>> in(height);
  for (int i = 0; i < height; i++) {
    in[i].resize(width);
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      in[i][j] = rand() % 256;
    }
  }

  std::vector<std::vector<int>> out(height);
  for (int i = 0; i < height; i++) {
    out[i].resize(width);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));

  // Create Task
  BlockFilterTBBTaskSequential blockFilterTBBTaskSequential(taskDataSeq);
  ASSERT_EQ(blockFilterTBBTaskSequential.validation(), true);
  blockFilterTBBTaskSequential.pre_processing();
  blockFilterTBBTaskSequential.run();
  blockFilterTBBTaskSequential.post_processing();

  std::vector<std::vector<int>> out_2(height);
  for (int i = 0; i < height; i++) {
    out_2[i].resize(width);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataPar->inputs_count.emplace_back(height);
  taskDataPar->inputs_count.emplace_back(width);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_2));

  // Create Task
  BlockFilterTBBTaskParallel blockFilterTBBTaskParallel(taskDataPar);
  ASSERT_EQ(blockFilterTBBTaskParallel.validation(), true);
  blockFilterTBBTaskParallel.pre_processing();
  blockFilterTBBTaskParallel.run();
  blockFilterTBBTaskParallel.post_processing();
  ASSERT_EQ(out[0][0], out_2[0][0]);
}

TEST(lapin_d_image_filter_block_parallel, Test_Lapin_test_200x200) {
  // Create data
  int height = 200;
  int width = 200;
  std::vector<std::vector<int>> in(height);
  for (int i = 0; i < height; i++) {
    in[i].resize(width);
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      in[i][j] = rand() % 256;
    }
  }

  std::vector<std::vector<int>> out(height);
  for (int i = 0; i < height; i++) {
    out[i].resize(width);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));

  // Create Task
  BlockFilterTBBTaskSequential blockFilterTBBTaskSequential(taskDataSeq);
  ASSERT_EQ(blockFilterTBBTaskSequential.validation(), true);
  blockFilterTBBTaskSequential.pre_processing();
  blockFilterTBBTaskSequential.run();
  blockFilterTBBTaskSequential.post_processing();

  std::vector<std::vector<int>> out_2(height);
  for (int i = 0; i < height; i++) {
    out_2[i].resize(width);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataPar->inputs_count.emplace_back(height);
  taskDataPar->inputs_count.emplace_back(width);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_2));

  // Create Task
  BlockFilterTBBTaskParallel blockFilterTBBTaskParallel(taskDataPar);
  ASSERT_EQ(blockFilterTBBTaskParallel.validation(), true);
  blockFilterTBBTaskParallel.pre_processing();
  blockFilterTBBTaskParallel.run();
  blockFilterTBBTaskParallel.post_processing();
  ASSERT_EQ(out[0][0], out_2[0][0]);
}

TEST(lapin_d_image_filter_block_parallel, Test_Lapin_test_300x300) {
  // Create data
  int height = 100;
  int width = 200;
  std::vector<std::vector<int>> in(height);
  for (int i = 0; i < height; i++) {
    in[i].resize(width);
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      in[i][j] = rand() % 256;
    }
  }

  std::vector<std::vector<int>> out(height);
  for (int i = 0; i < height; i++) {
    out[i].resize(width);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));

  // Create Task
  BlockFilterTBBTaskSequential blockFilterTBBTaskSequential(taskDataSeq);
  ASSERT_EQ(blockFilterTBBTaskSequential.validation(), true);
  blockFilterTBBTaskSequential.pre_processing();
  blockFilterTBBTaskSequential.run();
  blockFilterTBBTaskSequential.post_processing();

  std::vector<std::vector<int>> out_2(height);
  for (int i = 0; i < height; i++) {
    out_2[i].resize(width);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataPar->inputs_count.emplace_back(height);
  taskDataPar->inputs_count.emplace_back(width);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_2));

  // Create Task
  BlockFilterTBBTaskParallel blockFilterTBBTaskParallel(taskDataPar);
  ASSERT_EQ(blockFilterTBBTaskParallel.validation(), true);
  blockFilterTBBTaskParallel.pre_processing();
  blockFilterTBBTaskParallel.run();
  blockFilterTBBTaskParallel.post_processing();
  ASSERT_EQ(out[0][0], out_2[0][0]);
}