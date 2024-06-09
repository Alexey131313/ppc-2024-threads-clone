// Copyright 2024 Ustinov Alexander
#pragma once

#include <complex>
#include <random>
#include <vector>

// sparse matrix in CSC (Compressed Sparse Column) format
struct sparse_matrix {
  int row_num, col_num;                      // number of rows and columns in matrix
  int nonzeros;                              // number of non-zero elements
  std::vector<int> col_ptr;                  // index at which column data in `rows` and `values` begins
  std::vector<int> rows;                     // rows of non-zero elements of matrix
  std::vector<std::complex<double>> values;  // values of non-zero elements in matrix

  sparse_matrix(int row_num_ = 0, int col_num_ = 0, int nonzeros_ = 0)
      : row_num(row_num_),
        col_num(col_num_),
        nonzeros(nonzeros_),
        col_ptr(row_num + 1),
        rows(nonzeros),
        values(nonzeros) {}
};