// Copyright 2024 Kosarev Egor
#include <gtest/gtest.h>

#include <vector>

#include "seq/kosarev_e_jarvis/include/ops_seq.hpp"

TEST(Kosarev_e_jarvis_seq, Test_hull_one) {
  std::vector<Kosarev_e_JarvisHull::Point> points = {{0, 3}, {1, 1}, {2, 2}, {4, 4}, {0, 0}, {1, 2}, {3, 1}, {3, 3}};
  std::vector<Kosarev_e_JarvisHull::Point> hull = {{0, 0}, {0, 3}, {4, 4}, {3, 1}};
  std::vector<Kosarev_e_JarvisHull::Point> resHull(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataSeq->outputs_count.emplace_back(resHull.size());

  // Create Task
  Kosarev_e_JarvisHull::TestTaskSequentialKosarevJarvisHull testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(resHull[i], hull[i]);
  }
}

TEST(Kosarev_e_jarvis_seq, Test_hull_two) {
  std::vector<Kosarev_e_JarvisHull::Point> points = {{2, 5},  {7, 3},  {1, 1}, {-1, 4}, {-5, 4}, {-4, 1}, {-1, -2},
                                                     {2, -3}, {3, -1}, {2, 2}, {1, 5},  {-4, 5}, {-1, 0}};
  std::vector<Kosarev_e_JarvisHull::Point> hull = {
      {2, -3}, {-1, -2}, {-4, 1}, {-5, 4}, {-4, 5}, {2, 5}, {7, 3},
  };
  std::vector<Kosarev_e_JarvisHull::Point> resHull(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataSeq->outputs_count.emplace_back(resHull.size());

  // Create Task
  Kosarev_e_JarvisHull::TestTaskSequentialKosarevJarvisHull testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(resHull[i], hull[i]);
  }
}

TEST(Kosarev_e_jarvis_seq, Test_hull_three) {
  std::vector<Kosarev_e_JarvisHull::Point> points = {{2, 5}, {7, 3}, {1, 1}};
  std::vector<Kosarev_e_JarvisHull::Point> hull = {{1, 1}, {2, 5}, {7, 3}};
  std::vector<Kosarev_e_JarvisHull::Point> resHull(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataSeq->outputs_count.emplace_back(resHull.size());

  // Create Task
  Kosarev_e_JarvisHull::TestTaskSequentialKosarevJarvisHull testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(resHull[i], hull[i]);
  }
}

TEST(Kosarev_e_jarvis_seq, Test_hull_no_hull) {
  std::vector<Kosarev_e_JarvisHull::Point> points = {};
  std::vector<Kosarev_e_JarvisHull::Point> hull = {};
  std::vector<Kosarev_e_JarvisHull::Point> resHull(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataSeq->outputs_count.emplace_back(resHull.size());

  // Create Task
  Kosarev_e_JarvisHull::TestTaskSequentialKosarevJarvisHull testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(resHull[i], hull[i]);
  }
}

TEST(Kosarev_e_jarvis_seq, Test_hull_yes_hull) {
  std::vector<Kosarev_e_JarvisHull::Point> points = {{1, 4}, {1, 5}, {1, 6}, {1, 2}, {1, 3},
                                                     {2, 5}, {3, 4}, {0, 5}, {1, 7}};
  std::vector<Kosarev_e_JarvisHull::Point> hull = {{1, 2}, {0, 5}, {1, 7}, {3, 4}};
  std::vector<Kosarev_e_JarvisHull::Point> resHull(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataSeq->outputs_count.emplace_back(resHull.size());

  // Create Task
  Kosarev_e_JarvisHull::TestTaskSequentialKosarevJarvisHull testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(resHull[i], hull[i]);
  }
}