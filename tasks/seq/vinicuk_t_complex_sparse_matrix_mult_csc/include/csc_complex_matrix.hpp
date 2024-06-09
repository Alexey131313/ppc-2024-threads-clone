// Copyright 2024 Vinichuk Timofey
#pragma once

#include <complex>
#include <random>
#include <vector>

struct CSCComplexMatrix {
  std::vector<std::complex<double>> values;
  std::vector<size_t> row_indexes;
  std::vector<size_t> col_ptrs;

  size_t num_rows;
  size_t num_cols;

  // CSSComplexMatrix(size_t n_rows, size_t n_cols);

  CSCComplexMatrix(size_t _num_cols, size_t _num_rows) : num_rows(_num_rows), num_cols(_num_cols) {}
};
