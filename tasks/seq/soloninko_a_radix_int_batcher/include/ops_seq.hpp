// Copyright 2024 Soloninko Andrey
#pragma once

#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core/task/include/task.hpp"

class TaskSequentialBatcherSoloninko : public ppc::core::Task {
 public:
  explicit TaskSequentialBatcherSoloninko(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> in_vec, res, p1, p2;
};

inline std::vector<int> sol_GetRandVector(int size) {
  std::random_device rd;
  std::uniform_int_distribution<int> unif(0, 1000000);
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = unif(rd);
  }

  return vec;
}