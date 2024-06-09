// Copyright 2024 Vetoshnikova Ekaterina
#include <gtest/gtest.h>

#include <vector>

#include "seq/vetoshnikova_k_convex_hull_bin_image_components/include/ops_seq.hpp"

TEST(vetoshnikova_k_hull_bin_image_seq, Test1) {
  // Create data
  int h = 6;
  int w = 6;
  std::vector<int> out(50, 0);
  std::vector<uint8_t> in(
      {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  std::vector<int> hullTrue = {1, 1, 1, 2, 2, 2, 2, 1, -1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  ConstructingConvexHullSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < hullTrue.size(); i++) {
    ASSERT_EQ(hullTrue[i], out[i]);
  }
}

TEST(vetoshnikova_k_hull_bin_image_seq, Test2) {
  // Create data
  int h = 5;
  int w = 6;
  std::vector<int> out(50);
  std::vector<uint8_t> in({0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  std::vector<int> hullTrue = {2, 1, 1, 2, 2, 3, 3, 2, -1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  ConstructingConvexHullSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < hullTrue.size(); i++) {
    ASSERT_EQ(hullTrue[i], out[i]);
  }
}

TEST(vetoshnikova_k_hull_bin_image_seq, Test3) {
  // Create data
  int h = 6;
  int w = 6;
  std::vector<int> out(50);
  std::vector<uint8_t> in(
      {0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0});

  std::vector<int> hullTrue = {1, 1, 0, 2, 1, 3, -1, 3, 3, 4, 4, 4, 3, -1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  ConstructingConvexHullSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < hullTrue.size(); i++) {
    ASSERT_EQ(hullTrue[i], out[i]);
  }
}
TEST(vetoshnikova_k_hull_bin_image_seq, Test4) {
  // Create data
  int h = 5;
  int w = 6;
  std::vector<int> out(50);
  std::vector<uint8_t> in({0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  std::vector<int> hullTrue = {1, 1, 1, 2, 1, 3, 2, 3, 3, 2, 2, 1, -1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  ConstructingConvexHullSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < hullTrue.size(); i++) {
    ASSERT_EQ(hullTrue[i], out[i]);
  }
}

TEST(vetoshnikova_k_hull_bin_image_seq, Test5) {
  // Create data
  int h = 7;
  int w = 7;
  std::vector<int> out(50);
  std::vector<uint8_t> in({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1,
                           1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0});

  std::vector<int> hullTrue = {3, 2, 2, 3, 3, 5, 4, 4, 4, 3, -1, 5, 0, 5, 1, 6, 1, 6, 0, -1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  ConstructingConvexHullSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < hullTrue.size(); i++) {
    ASSERT_EQ(hullTrue[i], out[i]);
  }
}
