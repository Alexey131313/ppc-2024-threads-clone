// Copyright Andrey Karagodin 2024.
#include <gtest/gtest.h>

#include <iostream>
#include <seq/karagodin_a_dejkstra/include/dejkstra_seq.hpp>
#include <vector>

TEST(karagodin_a_dejkstra_seq, test_correctness) {
  int entryNode = 0;
  int destNode = 1;
  int expectedScore = 7;
  size_t size = 5;  // Because we have graph done
  std::vector<int> pathRes = {0, 1};
  std::pair<std::vector<int>, int> resultExpected(pathRes, expectedScore);
  std::pair<std::vector<int>, int> result;
  std::vector<std::vector<int>> graphMap = {
      {0, 7, 5, 0, 0}, {7, 0, 3, 5, 0}, {5, 3, 0, 0, 4}, {0, 5, 0, 0, 6}, {0, 0, 4, 6, 0}};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&entryNode));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&destNode));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&graphMap));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  // Create Task
  DejkstraTaskSequential dejkstra(taskDataSeq);
  ASSERT_EQ(dejkstra.validation(), true);
  dejkstra.pre_processing();
  dejkstra.run();
  dejkstra.post_processing();
  ASSERT_EQ(resultExpected.first, result.first);
  ASSERT_EQ(resultExpected.second, result.second);
}

TEST(karagodin_a_dejkstra_seq, test_hard_path) {
  int entryNode = 0;
  int destNode = 8;
  int expectedScore = 50;
  int size = 12;
  std::vector<int> pathRes = {0, 6, 5, 3, 8};
  std::pair<std::vector<int>, int> resultExpected(pathRes, expectedScore);
  std::pair<std::vector<int>, int> result;

  std::vector<std::vector<int>> graphMap = {{
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                                39,
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                            },
                                            {
                                                0,
                                                0,
                                                28,
                                                24,
                                                8,
                                                28,
                                                42,
                                                8,
                                                0,
                                                0,
                                                28,
                                                0,
                                            },
                                            {
                                                13,
                                                28,
                                                0,
                                                0,
                                                0,
                                                0,
                                                29,
                                                37,
                                                18,
                                                15,
                                                24,
                                                0,
                                            },
                                            {
                                                9,
                                                24,
                                                0,
                                                0,
                                                0,
                                                5,
                                                0,
                                                39,
                                                5,
                                                0,
                                                31,
                                                41,
                                            },
                                            {
                                                5,
                                                8,
                                                0,
                                                0,
                                                0,
                                                20,
                                                0,
                                                37,
                                                38,
                                                0,
                                                25,
                                                11,
                                            },
                                            {
                                                0,
                                                28,
                                                0,
                                                5,
                                                20,
                                                0,
                                                1,
                                                15,
                                                0,
                                                34,
                                                0,
                                                0,
                                            },
                                            {
                                                39,
                                                42,
                                                29,
                                                0,
                                                0,
                                                1,
                                                0,
                                                37,
                                                40,
                                                15,
                                                31,
                                                0,
                                            },
                                            {
                                                0,
                                                8,
                                                37,
                                                39,
                                                37,
                                                15,
                                                37,
                                                0,
                                                0,
                                                9,
                                                0,
                                                0,
                                            },
                                            {
                                                0,
                                                0,
                                                18,
                                                5,
                                                38,
                                                0,
                                                40,
                                                0,
                                                0,
                                                37,
                                                28,
                                                20,
                                            },
                                            {
                                                0,
                                                0,
                                                15,
                                                0,
                                                0,
                                                34,
                                                15,
                                                9,
                                                37,
                                                0,
                                                15,
                                                1,
                                            },
                                            {
                                                9,
                                                28,
                                                24,
                                                31,
                                                25,
                                                0,
                                                31,
                                                0,
                                                28,
                                                15,
                                                0,
                                                0,
                                            },
                                            {
                                                0,
                                                0,
                                                0,
                                                41,
                                                11,
                                                0,
                                                0,
                                                0,
                                                20,
                                                1,
                                                0,
                                                0,
                                            }};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&entryNode));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&destNode));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&graphMap));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  // Create Task
  DejkstraTaskSequential dejkstra(taskDataSeq);
  ASSERT_EQ(dejkstra.validation(), true);
  dejkstra.pre_processing();
  dejkstra.run();
  dejkstra.post_processing();
  ASSERT_EQ(resultExpected.first, result.first);
  ASSERT_EQ(resultExpected.second, result.second);
}

TEST(karagodin_a_dejkstra_seq, test_not_generating_sizes_lesser_than_2) {
  // Any numbers really. Checking if algorithm crashes on large data sample.
  std::vector<std::vector<int>> graphMap(2);
  ASSERT_ANY_THROW(graphMap = initGraphMapRandom(0));
}

TEST(karagodin_a_dejkstra_seq, test_big_size_10) {
  // Any numbers really. Checking if algorithm crashes on large data sample.
  int entryNode = 0;
  int destNode = 9;
  int size = 10;
  std::vector<int> pathRes = {0, 6, 5, 3, 8};
  std::pair<std::vector<int>, int> result;
  std::vector<std::vector<int>> graphMap;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&entryNode));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&destNode));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&graphMap));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  // Create Task
  DejkstraTaskSequential dejkstra(taskDataSeq);
  ASSERT_EQ(dejkstra.validation(), true);
  dejkstra.pre_processing();
  ASSERT_NO_FATAL_FAILURE(dejkstra.run());
  dejkstra.post_processing();
}

TEST(karagodin_a_dejkstra_seq, test_big_size_50) {
  // Any numbers really. Checking if algorithm crashes on large data sample.
  int entryNode = 0;
  int destNode = 48;
  int expectedScore = 50;
  int size = 50;
  std::vector<int> pathRes = {0};
  std::pair<std::vector<int>, int> resultExpected(pathRes, expectedScore);
  std::pair<std::vector<int>, int> result;
  std::vector<std::vector<int>> graphMap;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&entryNode));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&destNode));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&graphMap));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  // Create Task
  DejkstraTaskSequential dejkstra(taskDataSeq);
  ASSERT_EQ(dejkstra.validation(), true);
  dejkstra.pre_processing();
  ASSERT_NO_FATAL_FAILURE(dejkstra.run());
  dejkstra.post_processing();
}
