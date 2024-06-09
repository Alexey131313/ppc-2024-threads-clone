// Copyright 2024 Safarov Nurlan
#pragma once

#ifndef TASKS_TBB_SAFAROV_N_SPARSE_MATMULT_CRS_INCLUDE_SPARSE_MATMULT_CRS_TBB_HPP_
#define TASKS_TBB_SAFAROV_N_SPARSE_MATMULT_CRS_INCLUDE_SPARSE_MATMULT_CRS_TBB_HPP_

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

class SparseMatrixCRS {
 public:
  int numberOfColumns;
  int numberOfRows;

  std::vector<double> values;
  std::vector<int> columnIndexes;
  std::vector<int> pointers;

  SparseMatrixCRS(int _numberOfColumns, int _numberOfRows, const std::vector<double>& _values,
                  const std::vector<int>& _columnIndexes, const std::vector<int>& _pointers);
  explicit SparseMatrixCRS(int _numberOfColumns = 0, int _numberOfRows = 0);
  explicit SparseMatrixCRS(std::vector<std::vector<double>> matrix);
  bool operator==(const SparseMatrixCRS& matrix) const;
};

std::vector<std::vector<double>> fillTheMatrixWithZeros(int columns, int rows);
std::vector<std::vector<double>> createRandomMatrix(int columns, int rows, double perc);
std::vector<std::vector<double>> multiplyMatrices(std::vector<std::vector<double>> A,
                                                  std::vector<std::vector<double>> B);

class SparseMatrixMultiplicationCRS_TBB : public ppc::core::Task {
 public:
  explicit SparseMatrixMultiplicationCRS_TBB(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  SparseMatrixCRS *X{}, *Y{}, *Z{};
};

#endif  // TASKS_TBB_SAFAROV_N_SPARSE_MATMULT_CRS_INCLUDE_SPARSE_MATMULT_CRS_TBB_HPP_
