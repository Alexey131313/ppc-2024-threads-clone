// Copyright 2024 Mironov Ilya
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

class MatrixCRS {
 public:
  int N;
  int NZ;
  std::vector<double> Value;
  std::vector<int> Col;
  std::vector<int> RowIndex;

  explicit MatrixCRS(int n = 0, int nz = 0);
  MatrixCRS(const double* matrix, int n, int m, bool transpose = false);
};

class MironovISequential : public ppc::core::Task {
 public:
  explicit MironovISequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static void genrateSparseMatrix(double* matrix, int sz, double ro);
  static void genrateEdMatrix(double* matrix, int n);

 private:
  MatrixCRS A;
  MatrixCRS BT;
  MatrixCRS C;
  double* c_out{};
  int M{};
};
