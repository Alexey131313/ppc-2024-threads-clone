// Copyright 2024 Smirnova Daria
#pragma once

#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

struct crs_matrix {
  int n_rows{};
  int n_cols{};
  std::vector<std::complex<double>> non_zero_values{};
  std::vector<int> pointer{};
  std::vector<int> col_indexes{};
};

class TestComplexMatrixCrsSeq : public ppc::core::Task {
 public:
  explicit TestComplexMatrixCrsSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  crs_matrix *A_M{}, *B_M{}, *Result{};
};
