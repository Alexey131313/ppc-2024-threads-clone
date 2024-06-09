// Copyright 2024 Alexseev Danila
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace alexseev_seq {
class HoareSortWBatcherMergeSequential : public ppc::core::Task {
 public:
  explicit HoareSortWBatcherMergeSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static void HoareSortWBatcherMergeSeq(std::vector<int> &arr, size_t l, size_t r);
  static void CompExch(int &a, int &b);

 private:
  std::vector<int> array{};
};
}  // namespace alexseev_seq