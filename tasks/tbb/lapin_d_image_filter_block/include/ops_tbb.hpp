// Copyright 2024 Lapin Dmitriy
#pragma once
#include <vector>

#include "core/task/include/task.hpp"

class BlockFilterTBBTaskSequential : public ppc::core::Task {
 public:
  explicit BlockFilterTBBTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int width;
  int height;
  std::vector<std::vector<int>> *mas_in;
  std::vector<std::vector<int>> *mas_out;
};

class BlockFilterTBBTaskParallel : public ppc::core::Task {
 public:
  explicit BlockFilterTBBTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int width;
  int height;
  std::vector<std::vector<int>> *mas_in;
  std::vector<std::vector<int>> *mas_out;
};