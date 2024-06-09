// Copyright 2024 Bodrov Daniil
#pragma once

#include <complex>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

struct SparseMatrix {
  int n_rows{};
  int n_cols{};
  std::vector<std::complex<double>> non_zero_values{};
  std::vector<int> pointer{};
  std::vector<int> col_indexes{};
};

class SparseMatrixSolver : public ppc::core::Task {
 public:
  explicit SparseMatrixSolver(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  SparseMatrix *A_M{}, *B_M{}, *Result{};
};
