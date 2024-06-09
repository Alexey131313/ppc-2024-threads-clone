// Copyright 2024 Vetoshnikova Ekaterina
#include <gtest/gtest.h>

#include <vector>

#include "omp/vetoshnikova_k_convex_hull_bin_image_components/include/ops_omp.hpp"

TEST(vetoshnikova_k_hull_bin_image_omp, Test1) {
  // Create data
  int h = 6;
  int w = 6;
  std::vector<int> out_ref(50, 0);
  std::vector<uint8_t> in(
      {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  std::vector<int> hullTrue = {1, 1, 1, 2, 2, 2, 2, 1, -1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.data()));

  // Create Task
  vetoshnikova_omp::ConstructingConvexHullSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create data
  std::vector<int> out_par(50, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(h);
  taskDataPar->inputs_count.emplace_back(w);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));

  // Create Task
  vetoshnikova_omp::ConstructingConvexHullOMP testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  for (size_t i = 0; i < out_ref.size(); i++) {
    ASSERT_EQ(out_ref[i], out_par[i]);
  }
}

TEST(vetoshnikova_k_hull_bin_image_omp, Test2) {
  // Create data
  int h = 5;
  int w = 6;
  std::vector<int> out_ref(50);
  std::vector<uint8_t> in({0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  std::vector<int> hullTrue = {2, 1, 1, 2, 2, 3, 3, 2, -1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.data()));

  // Create Task
  vetoshnikova_omp::ConstructingConvexHullSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create data
  std::vector<int> out_par(50, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(h);
  taskDataPar->inputs_count.emplace_back(w);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));

  // Create Task
  vetoshnikova_omp::ConstructingConvexHullOMP testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  for (size_t i = 0; i < out_ref.size(); i++) {
    ASSERT_EQ(out_ref[i], out_par[i]);
  }
}

TEST(vetoshnikova_k_hull_bin_image_omp, Test3) {
  // Create data
  int h = 6;
  int w = 6;
  std::vector<int> out_ref(50);
  std::vector<uint8_t> in(
      {0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0});

  std::vector<int> hullTrue = {1, 1, 0, 2, 1, 3, -1, 3, 3, 4, 4, 4, 3, -1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.data()));

  // Create Task
  vetoshnikova_omp::ConstructingConvexHullSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create data
  std::vector<int> out_par(50, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(h);
  taskDataPar->inputs_count.emplace_back(w);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));

  // Create Task
  vetoshnikova_omp::ConstructingConvexHullOMP testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  for (size_t i = 0; i < out_ref.size(); i++) {
    ASSERT_EQ(out_ref[i], out_par[i]);
  }
}
TEST(vetoshnikova_k_hull_bin_image_omp, Test4) {
  // Create data
  int h = 5;
  int w = 6;
  std::vector<int> out_ref(50);
  std::vector<uint8_t> in({0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  std::vector<int> hullTrue = {1, 1, 1, 2, 1, 3, 2, 3, 3, 2, 2, 1, -1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.data()));

  // Create Task
  vetoshnikova_omp::ConstructingConvexHullSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create data
  std::vector<int> out_par(50, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(h);
  taskDataPar->inputs_count.emplace_back(w);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));

  // Create Task
  vetoshnikova_omp::ConstructingConvexHullOMP testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  for (size_t i = 0; i < out_ref.size(); i++) {
    ASSERT_EQ(out_ref[i], out_par[i]);
  }
}

TEST(vetoshnikova_k_hull_bin_image_omp, Test5) {
  // Create data
  int h = 7;
  int w = 7;
  std::vector<int> out_ref(50);
  std::vector<uint8_t> in({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1,
                           1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0});

  std::vector<int> hullTrue = {3, 2, 2, 3, 3, 5, 4, 4, 4, 3, -1, 5, 0, 5, 1, 6, 1, 6, 0, -1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.data()));

  // Create Task
  vetoshnikova_omp::ConstructingConvexHullSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create data
  std::vector<int> out_par(50, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(h);
  taskDataPar->inputs_count.emplace_back(w);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));

  // Create Task
  vetoshnikova_omp::ConstructingConvexHullOMP testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  for (size_t i = 0; i < out_ref.size(); i++) {
    ASSERT_EQ(out_ref[i], out_par[i]);
  }
}
