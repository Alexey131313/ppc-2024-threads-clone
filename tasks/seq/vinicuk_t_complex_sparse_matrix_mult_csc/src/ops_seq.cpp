// Copyright 2024 Vinichuk Timofey

#include "seq/vinicuk_t_complex_sparse_matrix_mult_csc/include/ops_seq.hpp"

#include <algorithm>
#include <complex>
#include <iostream>

bool MultMatrixCSCComplex::pre_processing() {
  internal_order_test();

  mtrx_A = reinterpret_cast<CSCComplexMatrix*>(taskData->inputs[0]);
  mtrx_B = reinterpret_cast<CSCComplexMatrix*>(taskData->inputs[1]);
  mtrx_res = reinterpret_cast<CSCComplexMatrix*>(taskData->outputs[0]);
  return true;
}

bool MultMatrixCSCComplex::validation() {
  internal_order_test();

  int res_col_num = reinterpret_cast<CSCComplexMatrix*>(taskData->outputs[0])->num_cols;
  int res_row_num = reinterpret_cast<CSCComplexMatrix*>(taskData->outputs[0])->num_rows;
  int A_col_num = reinterpret_cast<CSCComplexMatrix*>(taskData->inputs[0])->num_cols;
  int A_row_num = reinterpret_cast<CSCComplexMatrix*>(taskData->inputs[0])->num_rows;
  int B_col_num = reinterpret_cast<CSCComplexMatrix*>(taskData->inputs[1])->num_cols;
  int B_row_num = reinterpret_cast<CSCComplexMatrix*>(taskData->inputs[1])->num_rows;

  return A_col_num == B_row_num && res_col_num == B_col_num && res_row_num == A_row_num;
}

bool MultMatrixCSCComplex::run() {
  internal_order_test();

  std::vector<std::complex<double>> values;
  std::vector<size_t> row_indexes;
  std::vector<size_t> col_ptrs;
  std::complex<double> sum;
  mtrx_res->col_ptrs.push_back(0);
  for (size_t j = 0; j < mtrx_B->num_cols; j++) {
    for (size_t i = 0; i < mtrx_A->num_rows; i++) {
      // if (mtrx_B.col_ptrs[j] - mtrx_B.col_ptrs[j+1] != 0)
      sum.imag(0.0);
      sum.real(0.0);
      for (size_t k = mtrx_B->col_ptrs[j]; k < mtrx_B->col_ptrs[j + 1]; k++) {
        auto start = mtrx_A->row_indexes.begin() + mtrx_A->col_ptrs[mtrx_B->row_indexes[k]];
        auto end = mtrx_A->row_indexes.begin() + mtrx_A->col_ptrs[mtrx_B->row_indexes[k] + 1];
        auto it = find(start, end, i);
        if (it != end) {
          int index = distance(mtrx_A->row_indexes.begin(), it);
          sum += mtrx_A->values[index] * mtrx_B->values[k];
        }
      }
      if (sum.imag() != 0.0 || sum.real() != 0.0) {
        mtrx_res->values.push_back(sum);
        mtrx_res->row_indexes.push_back(i);
      }
    }
    mtrx_res->col_ptrs.push_back(mtrx_res->values.size());
  }

  return true;
}

bool MultMatrixCSCComplex::post_processing() {
  internal_order_test();
  return true;
}
