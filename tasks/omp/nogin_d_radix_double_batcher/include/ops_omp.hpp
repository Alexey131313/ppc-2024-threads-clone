// Copyright 2024 Nogin Denis
#pragma once

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace NoginDenisOmp {

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

class RadixSortDoubleBatcherOmpParallel : public ppc::core::Task {
 public:
  explicit RadixSortDoubleBatcherOmpParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> arr;
  std::vector<double> res;
};

std::vector<double> batchersMergeSeq(std::vector<std::vector<double>>& subvectors);
void partSortSeq(std::vector<std::vector<double>>& parts, std::vector<double>& side);
std::vector<double> radixSortBatcherSeq(std::vector<double> vec);

std::vector<double> batchersMergeOmp(std::vector<std::vector<double>>& subvectors);
void partSortOmp(std::vector<std::vector<double>>& parts, std::vector<double>& side);
std::vector<double> radixSortBatcherOmp(std::vector<double> vec);

std::vector<double> randomVector(int sizeVec, double minValue, double maxValue);
}  // namespace NoginDenisOmp
