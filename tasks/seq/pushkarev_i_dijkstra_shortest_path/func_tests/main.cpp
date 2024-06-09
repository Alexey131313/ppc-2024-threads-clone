// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/pushkarev_i_dijkstra_shortest_path/include/dijkstra_shortest_path.hpp"

TEST(pushkarev_i_dijkstra_shortest_path_seq, Test_validation_fail) {
  const int count = 4;

  // Create data
  std::vector<std::vector<int>> graph{{0, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 1}, {1, 1, -1, 0}};
  std::vector<int> distances(count, 0);
  int source = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&graph));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source));
  taskDataSeq->inputs_count.emplace_back(2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&distances));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  DijkstraTask dijkstraTask(taskDataSeq);

  ASSERT_EQ(dijkstraTask.validation(), false);
}

TEST(pushkarev_i_dijkstra_shortest_path_seq, Test_chain_graph) {
  const int count = 4;

  // Create data
  std::vector<std::vector<int>> graph{{0, 1, 9999, 9999}, {1, 0, 1, 9999}, {9999, 1, 0, 1}, {9999, 9999, 1, 0}};
  std::vector<int> distances(count, 0);
  int source = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&graph));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source));
  taskDataSeq->inputs_count.emplace_back(2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&distances));
  taskDataSeq->outputs_count.emplace_back(1);

  DijkstraTask dijkstraTask(taskDataSeq);

  // Create Task
  dijkstraTask.validation();
  dijkstraTask.pre_processing();
  dijkstraTask.run();
  dijkstraTask.post_processing();

  ASSERT_EQ(distances[0], 0);
  ASSERT_EQ(distances[1], 1);
  ASSERT_EQ(distances[2], 2);
  ASSERT_EQ(distances[3], 3);
}

TEST(pushkarev_i_dijkstra_shortest_path_seq, Test_trivial_case_N4) {
  const int count = 4;

  // Create data
  std::vector<std::vector<int>> graph{{0, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 1}, {1, 1, 1, 0}};
  std::vector<int> distances(count, 0);
  int source = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&graph));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source));
  taskDataSeq->inputs_count.emplace_back(2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&distances));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  DijkstraTask dijkstraTask(taskDataSeq);

  dijkstraTask.validation();
  dijkstraTask.pre_processing();
  dijkstraTask.run();
  dijkstraTask.post_processing();

  ASSERT_EQ(distances[0], 0);
  ASSERT_EQ(distances[1], 1);
  ASSERT_EQ(distances[2], 1);
  ASSERT_EQ(distances[3], 1);
}

TEST(pushkarev_i_dijkstra_shortest_path_seq, Test_random_graph_N6) {
  const int count = 6;

  // Create data
  std::vector<std::vector<int>> graph{
      {0, 10, 2, 8, 9999, 9999}, {10, 0, 5, 9999, 9999, 9999}, {2, 5, 0, 3, 1, 6},
      {8, 9999, 3, 0, 6, 9999},  {9999, 9999, 1, 6, 0, 3},     {9999, 9999, 6, 9999, 3, 0},

  };
  std::vector<int> distances(count, 0);
  int source = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&graph));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source));
  taskDataSeq->inputs_count.emplace_back(2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&distances));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  DijkstraTask dijkstraTask(taskDataSeq);

  dijkstraTask.validation();
  dijkstraTask.pre_processing();
  dijkstraTask.run();
  dijkstraTask.post_processing();

  ASSERT_EQ(distances[0], 0);
  ASSERT_EQ(distances[1], 7);
  ASSERT_EQ(distances[2], 2);
  ASSERT_EQ(distances[3], 5);
  ASSERT_EQ(distances[4], 3);
  ASSERT_EQ(distances[5], 6);
}

TEST(pushkarev_i_dijkstra_shortest_path_seq, Test_full_graph_N6) {
  const int count = 6;

  // Create data
  std::vector<std::vector<int>> graph{
      {0, 3, 4, 5, 6, 7}, {3, 0, 2, 2, 2, 1},  {4, 2, 0, 3, 13, 2},
      {5, 2, 3, 0, 1, 0}, {6, 2, 13, 1, 0, 2}, {7, 1, 2, 0, 2, 0},

  };
  std::vector<int> distances(count, 0);
  int source = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&graph));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source));
  taskDataSeq->inputs_count.emplace_back(2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&distances));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  DijkstraTask dijkstraTask(taskDataSeq);

  dijkstraTask.validation();
  dijkstraTask.pre_processing();
  dijkstraTask.run();
  dijkstraTask.post_processing();

  ASSERT_EQ(distances[0], 0);
  ASSERT_EQ(distances[1], 3);
  ASSERT_EQ(distances[2], 4);
  ASSERT_EQ(distances[3], 5);
  ASSERT_EQ(distances[4], 5);
  ASSERT_EQ(distances[5], 4);
}