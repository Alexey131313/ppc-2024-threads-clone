// Copyright 2024 Moiseev Nikita
#include <gtest/gtest.h>

#include <vector>

#include "omp/moiseev_n_jarvis/include/ops_omp.hpp"

TEST(Moiseev_n_jarvis_omp, Empty_Input) {
  std::vector<Point> points = {};
  std::vector<Point> resHull_seq(points.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull_seq.data()));
  taskDataSeq->outputs_count.emplace_back(resHull_seq.size());

  // Create Task
  TestTaskSequentialJarvis testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);

  // Create data
  std::vector<Point> resHull_par(points.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataPar->inputs_count.emplace_back(points.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull_par.data()));
  taskDataPar->outputs_count.emplace_back(resHull_par.size());

  // Create Task
  TestOMPTaskParallelMoiseevJarvis testOmpTaskParallel(taskDataPar);
  ASSERT_EQ(testOmpTaskParallel.validation(), false);
}

TEST(Moiseev_n_jarvis_omp, Single_Point) {
  std::vector<Point> points = {{0, 0}};
  std::vector<Point> resHull_seq(points.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull_seq.data()));
  taskDataSeq->outputs_count.emplace_back(resHull_seq.size());

  // Create Task
  TestTaskSequentialJarvis testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create data
  std::vector<Point> resHull_par(points.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataPar->inputs_count.emplace_back(points.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull_par.data()));
  taskDataPar->outputs_count.emplace_back(resHull_par.size());

  // Create Task
  TestOMPTaskParallelMoiseevJarvis testOmpTaskParallel(taskDataPar);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  for (size_t i = 0; i < resHull_par.size(); ++i) {
    ASSERT_EQ(resHull_par[i], resHull_seq[i]);
  }
}

TEST(Moiseev_n_jarvis_omp, Two_Points) {
  std::vector<Point> points = {{0, 0}, {1, 1}};
  std::vector<Point> resHull_seq(points.size());

  // Create TaskData for sequential version
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull_seq.data()));
  taskDataSeq->outputs_count.emplace_back(resHull_seq.size());

  // Create Task for sequential version
  TestTaskSequentialJarvis testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create data for parallel version
  std::vector<Point> resHull_par(points.size());

  // Create TaskData for parallel version
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataPar->inputs_count.emplace_back(points.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull_par.data()));
  taskDataPar->outputs_count.emplace_back(resHull_par.size());

  // Create Task for parallel version
  TestOMPTaskParallelMoiseevJarvis testOmpTaskParallel(taskDataPar);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  for (size_t i = 0; i < resHull_par.size(); ++i) {
    ASSERT_EQ(resHull_par[i], resHull_seq[i]);
  }
}

TEST(Moiseev_n_jarvis_omp, Duplicate_Points) {
  std::vector<Point> points = {{0, 0}, {1, 1}, {2, 2}, {0, 0}, {1, 1}, {2, 2}};
  std::vector<Point> resHull_seq(2);

  // Create TaskData for sequential version
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull_seq.data()));
  taskDataSeq->outputs_count.emplace_back(resHull_seq.size());

  // Create Task for sequential version
  TestTaskSequentialJarvis testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create data for parallel version
  std::vector<Point> resHull_par(2);

  // Create TaskData for parallel version
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataPar->inputs_count.emplace_back(points.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull_par.data()));
  taskDataPar->outputs_count.emplace_back(resHull_par.size());

  // Create Task for parallel version
  TestOMPTaskParallelMoiseevJarvis testOmpTaskParallel(taskDataPar);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  for (size_t i = 0; i < resHull_par.size(); ++i) {
    ASSERT_EQ(resHull_par[i], resHull_seq[i]);
  }
}

TEST(Moiseev_n_jarvis_omp, Random_Points) {
  std::vector<Point> points = {{1, 4}, {3, 8}, {8, 2}, {5, 5}, {9, 1}, {4, 7}};
  std::vector<Point> hull = {{1, 4}, {3, 8}, {4, 7}, {9, 1}};
  std::vector<Point> resHull(hull.size());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataSeq->outputs_count.emplace_back(resHull.size());

  // Create Task
  TestTaskSequentialJarvis testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create OMP TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataOmp->inputs_count.emplace_back(points.size());
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataOmp->outputs_count.emplace_back(resHull.size());

  // Create OMP Task
  TestOMPTaskParallelMoiseevJarvis testOmpTaskParallel(taskDataOmp);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(resHull[i], hull[i]);
  }
}

TEST(Moiseev_n_jarvis_omp, Rectangle_Points) {
  std::vector<Point> points = {{0, 0}, {0, 5}, {5, 5}, {5, 0}, {1, 1}, {1, 4}, {4, 4}, {4, 1}};
  std::vector<Point> hull = {{0, 0}, {0, 5}, {5, 5}, {5, 0}};
  std::vector<Point> resHull(hull.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataSeq->outputs_count.emplace_back(resHull.size());

  // Create Task
  TestTaskSequentialJarvis testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create OMP TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataOmp->inputs_count.emplace_back(points.size());
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(resHull.data()));
  taskDataOmp->outputs_count.emplace_back(resHull.size());

  // Create OMP Task
  TestOMPTaskParallelMoiseevJarvis testOmpTaskParallel(taskDataOmp);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(resHull[i], hull[i]);
  }
}
