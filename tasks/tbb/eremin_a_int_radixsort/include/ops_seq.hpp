// Copyright 2024 Eremin Alexander
#pragma once

#include <tbb/tbb.h>

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class RadixSortTaskTBB : public ppc::core::Task {
 public:
  explicit RadixSortTaskTBB(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> VectorForSort;
  int SizeVector = 0;
};

class RadixSortBody {
 public:
  mutable std::vector<int> VectorResult;
  RadixSortBody() = default;
  RadixSortBody(const RadixSortBody&, tbb::split) {}

  void operator()(const tbb::blocked_range<std::vector<int>::iterator>& r) {
    std::vector<int> tmp(r.begin(), r.end());
    tmp = radixSort(tmp);
    VectorResult = myMerge(&tmp, &VectorResult);
  }

  void join(RadixSortBody& rhs) { VectorResult = myMerge(&rhs.VectorResult, &VectorResult); }

  static std::vector<int> radixSort(std::vector<int> vec);
  static std::vector<int> myMerge(std::vector<int>* OneVector, std::vector<int>* TwoVector);
};
