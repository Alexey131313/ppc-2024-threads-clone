// Copyright 2024 Polozov Vladislav
#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

class SortHoarWithMergeBatcher : public ppc::core::Task {
 public:
  explicit SortHoarWithMergeBatcher(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> arr;
  std::vector<int> res;
};

int partition(std::vector<int>& a, int l, int r);
void Hoar_sort(std::vector<int>& a, int l, int r);
std::vector<int> generate_data(int n, int mn, int mx);
void CompAndSwap(int& x, int& y);
std::vector<int> odd_even_merge_with_hoar(std::vector<int>& my_data);