// Copyright 2023 Pushkarev Ilya
#pragma once

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

class DijkstraTask : public ppc::core::Task {
 public:
  explicit DijkstraTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> distances_;

  std::vector<std::vector<int>> graph;
  int source;

  void initializeDistances();
  size_t getMinDistanceVertex(const std::vector<bool>& processed);
  void relaxVertex(size_t u, size_t v);
};
