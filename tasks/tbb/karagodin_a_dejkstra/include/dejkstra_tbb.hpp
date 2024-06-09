// Copyright 2024 Karagodin Andrey
#pragma once

#ifndef TASKS_TBB_KARAGODIN_A_DEJKSTRA_INCLUDE_DEJKSTRA_TBB_HPP_
#define TASKS_TBB_KARAGODIN_A_DEJKSTRA_INCLUDE_DEJKSTRA_TBB_HPP_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

class DejkstraTaskTBBSequential final : public ppc::core::Task {
 public:
  explicit DejkstraTaskTBBSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::pair<std::vector<int>, int> getDejMinPath();
  struct Node {
    int vertex;
    int cost;
    Node(const int v, const int c) : vertex(v), cost(c) {}
  };

  struct CompareNode {
    bool operator()(const Node& n1, const Node& n2) const { return n1.cost > n2.cost; }
  };
  static void printGraphMap(const std::vector<std::vector<int>>& graphMapInput);

 private:
  int size = 0;
  int destNode = 0;
  int entryNode = 0;
  int minScore = 0;
  std::vector<int> pathOutput;
  std::vector<std::vector<int>> graphMap;
  std::pair<std::vector<int>, int> res;
};

class DejkstraTaskTBB final : public ppc::core::Task {
 public:
  explicit DejkstraTaskTBB(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::pair<std::vector<int>, int> getDejMinPath();
  struct Node {
    int vertex;
    int cost;
    Node(const int v, const int c) : vertex(v), cost(c) {}
  };

  struct CompareNode {
    bool operator()(const Node& n1, const Node& n2) const { return n1.cost > n2.cost; }
  };
  static void printGraphMap(const std::vector<std::vector<int>>& graphMapInput);

 private:
  int size = 0;
  int destNode = 0;
  int entryNode = 0;
  int minScore = 0;
  std::vector<int> pathOutput;
  std::vector<std::vector<int>> graphMap;
  std::pair<std::vector<int>, int> res;
};

std::vector<std::vector<int>> initGraphMapRandom(int16_t size);

#endif  // TASKS_TBB_KARAGODIN_A_DEJKSTRA_INCLUDE_DEJKSTRA_TBB_HPP_