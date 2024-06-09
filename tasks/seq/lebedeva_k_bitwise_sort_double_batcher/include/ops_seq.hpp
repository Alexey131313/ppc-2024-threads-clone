// Copyright Lebedeva_Ksenia 2024
#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

class RadixSortDoubleBatcherSequential : public ppc::core::Task {
 public:
  explicit RadixSortDoubleBatcherSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> arr;
  std::vector<double> res;
};

std::vector<double> batchersMerge(std::vector<std::vector<double>>& subvectors);
void partSort(std::vector<std::vector<double>>& parts, std::vector<double>& side);
std::vector<double> bitwise_sort_batcher(std::vector<double> v);
std::vector<double> randomVector(int sizeVec, double minValue, double maxValue);
