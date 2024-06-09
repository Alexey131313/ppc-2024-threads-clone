// Copyright 2024 Alexseev Danila
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace alexseev_omp {
class HoareSortWBatcherMergeSequential : public ppc::core::Task {
 public:
  explicit HoareSortWBatcherMergeSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static void HoareSortWBatcherMergeSeq(std::vector<int> &arr, size_t l, size_t r);

 private:
  std::vector<int> array{};
};

class HoareSortWBatcherMergeOMP : public ppc::core::Task {
 public:
  explicit HoareSortWBatcherMergeOMP(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static void HoareSortWBatcherMergeParallel(std::vector<int> &arr, size_t l, size_t r);

 private:
  std::vector<int> array{};
};

void CompExch(int &a, int &b);
}  // namespace alexseev_omp