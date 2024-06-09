// Copyright 2024 Prokofev Kirill
#include <gtest/gtest.h>

#include <vector>

#include "seq/prokofev_k_convexHullForBinImage/include/ops_seq.hpp"

TEST(prokofev_k_convex_hull_seq, BinarImgConvexHullTest1) {
  // Create data
  std::vector<int> out(100);
  std::vector<std::vector<int>> image({{0, 1, 0, 0, 0, 1, 1, 1},
                                       {1, 1, 0, 0, 1, 1, 1, 1},
                                       {1, 1, 0, 1, 1, 1, 1, 1},
                                       {1, 0, 0, 1, 0, 0, 0, 1},
                                       {1, 1, 0, 0, 0, 0, 1, 1}});
  int width = 8;
  int height = 5;
  std::vector<int> trueRes = {0, 1, 1, 0, 1, 4, 0, 4, -1, 3, 2, 4, 1, 7, 0, 7, 4, 6, 4, 3, 3, -1};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&image));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  BinaryImageConvexHullSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < trueRes.size(); i++) {
    ASSERT_EQ(trueRes[i], out[i]);
  }
}

TEST(prokofev_k_convex_hull_seq, BinarImgConvexHullTest2) {
  // Create data
  std::vector<int> out(100);
  std::vector<std::vector<int>> image({{1, 1, 1, 1, 1, 1, 1, 1},
                                       {1, 0, 0, 0, 0, 0, 0, 1},
                                       {1, 0, 1, 0, 1, 1, 0, 1},
                                       {1, 0, 1, 0, 0, 0, 0, 1},
                                       {1, 0, 0, 1, 0, 1, 0, 1},
                                       {1, 0, 1, 0, 1, 1, 0, 1},
                                       {1, 0, 0, 0, 0, 0, 0, 1},
                                       {1, 1, 1, 1, 1, 1, 1, 1}});
  int width = 8;
  int height = 8;
  std::vector<int> trueRes = {0, 0, 7, 0, 7, 7, 0, 7, -1, 2, 2, 5, 4, 5, 5, 4, 5, 2, 5, -1, 4, 2, 5, 2, -1};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&image));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  BinaryImageConvexHullSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < trueRes.size(); i++) {
    ASSERT_EQ(trueRes[i], out[i]);
  }
}

TEST(prokofev_k_convex_hull_seq, BinarImgConvexHullTest3) {
  // Create data
  std::vector<int> out(100);
  std::vector<std::vector<int>> image({{1, 1, 0, 0, 0, 0, 1, 1},
                                       {1, 1, 0, 0, 0, 0, 1, 1},
                                       {1, 1, 0, 0, 0, 0, 1, 1},
                                       {1, 1, 0, 0, 0, 0, 1, 1},
                                       {1, 1, 0, 0, 0, 0, 1, 1}});
  int width = 8;
  int height = 5;
  std::vector<int> trueRes = {0, 0, 1, 0, 1, 4, 0, 4, -1, 6, 0, 7, 0, 7, 4, 6, 4, -1};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&image));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  BinaryImageConvexHullSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < trueRes.size(); i++) {
    ASSERT_EQ(trueRes[i], out[i]);
  }
}
TEST(prokofev_k_convex_hull_seq, BinarImgConvexHullTest4) {
  // Create data
  std::vector<int> out(100);
  std::vector<std::vector<int>> image({{0, 0, 0, 0, 0, 0, 0, 0, 0},
                                       {0, 0, 0, 0, 1, 0, 0, 0, 0},
                                       {0, 0, 0, 1, 0, 1, 0, 0, 0},
                                       {0, 0, 1, 0, 0, 0, 1, 0, 0},
                                       {0, 0, 0, 0, 0, 0, 0, 0, 0}});
  int width = 9;
  int height = 5;
  std::vector<int> trueRes = {2, 3, 3, 2, 5, 2, 6, 3, -1};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&image));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  BinaryImageConvexHullSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < trueRes.size(); i++) {
    ASSERT_EQ(trueRes[i], out[i]);
  }
}

TEST(prokofev_k_convex_hull_seq, BinarImgConvexHullTest5) {
  // Create data
  std::vector<int> out(100);
  std::vector<std::vector<int>> image({{0, 0, 1, 1, 1, 1, 0, 0},
                                       {0, 0, 1, 1, 1, 1, 0, 0},
                                       {0, 0, 1, 1, 1, 1, 0, 0},
                                       {0, 0, 1, 1, 1, 1, 0, 0},
                                       {0, 0, 1, 1, 1, 1, 0, 0},
                                       {0, 0, 1, 1, 1, 1, 0, 0},
                                       {0, 0, 1, 1, 1, 1, 0, 0},
                                       {0, 0, 1, 1, 1, 1, 0, 0},
                                       {0, 0, 1, 1, 1, 1, 0, 0},
                                       {0, 0, 1, 1, 1, 1, 0, 0}});
  int width = 8;
  int height = 10;
  std::vector<int> trueRes = {2, 0, 5, 0, 5, 9, 2, 9, -1};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&image));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  BinaryImageConvexHullSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < trueRes.size(); i++) {
    ASSERT_EQ(trueRes[i], out[i]);
  }
}