// Copyright 2024 Karagodin Andrey
#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

class DejkstraTaskSequential : public ppc::core::Task {
 public:
  explicit DejkstraTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::pair<std::vector<int>, int> getDejMinPath();
  struct Node {
    int vertex;
    int cost;
    Node(int v, int c) : vertex(v), cost(c) {}
  };

  struct CompareNode {
    bool operator()(const Node& n1, const Node& n2) { return n1.cost > n2.cost; }
  };
  static void printGraphMap(const std::vector<std::vector<int>>& graphMapInput);

 private:
  int size, destNode, entryNode, minScore;
  std::vector<int> pathOutput;
  std::vector<std::vector<int>> graphMap;
  std::pair<std::vector<int>, int> res;
};

std::vector<std::vector<int>> initGraphMapRandom(int16_t size);
