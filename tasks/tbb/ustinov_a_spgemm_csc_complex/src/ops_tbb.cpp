// Copyright 2024 Ustinov Alexander
#include "tbb/ustinov_a_spgemm_csc_complex/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_for.h>

#include <iostream>

bool SpgemmCSCComplexTBBSeq::pre_processing() {
  internal_order_test();

  A = reinterpret_cast<sparse_matrix *>(taskData->inputs[0]);
  B = reinterpret_cast<sparse_matrix *>(taskData->inputs[1]);
  C = reinterpret_cast<sparse_matrix *>(taskData->outputs[0]);
  return true;
}

bool SpgemmCSCComplexTBBSeq::validation() {
  internal_order_test();

  int A_col_num = reinterpret_cast<sparse_matrix *>(taskData->inputs[0])->col_num;
  int B_row_num = reinterpret_cast<sparse_matrix *>(taskData->inputs[1])->row_num;
  // check that matrices are compatible for multiplication
  return (A_col_num == B_row_num);
}

bool SpgemmCSCComplexTBBSeq::run() {
  internal_order_test();

  // symbolic stage
  C->row_num = A->row_num;
  C->col_num = B->col_num;
  C->col_ptr.resize(C->col_num + 1);
  C->col_ptr[0] = 0;
  std::vector<int> present_elements(C->row_num);
  for (int b_col = 0; b_col < C->col_num; ++b_col) {
    for (int c_row = 0; c_row < C->row_num; ++c_row) {
      present_elements[c_row] = 0;
    }
    for (int b_idx = B->col_ptr[b_col]; b_idx < B->col_ptr[b_col + 1]; ++b_idx) {
      int b_row = B->rows[b_idx];
      for (int a_idx = A->col_ptr[b_row]; a_idx < A->col_ptr[b_row + 1]; ++a_idx) {
        present_elements[A->rows[a_idx]] = 1;
      }
    }
    int col_nonzero_count = 0;
    for (int c_row = 0; c_row < C->row_num; ++c_row) {
      col_nonzero_count += present_elements[c_row];
    }
    C->col_ptr[b_col + 1] = col_nonzero_count + C->col_ptr[b_col];
  }

  // allocate memory for matrix C
  int total_nonzeros = C->col_ptr[C->col_num];
  C->nonzeros = total_nonzeros;
  C->rows.resize(total_nonzeros);
  C->values.resize(total_nonzeros);

  // numeric stage
  std::complex<double> zero;
  std::complex<double> b_value;
  std::vector<std::complex<double>> accumulator(C->row_num);
  for (int b_col = 0; b_col < C->col_num; ++b_col) {
    // set accumulator values to zero
    for (int c_row = 0; c_row < C->row_num; ++c_row) {
      accumulator[c_row] = zero;
      present_elements[c_row] = 0;
    }
    // calculate column into accumulator
    for (int b_idx = B->col_ptr[b_col]; b_idx < B->col_ptr[b_col + 1]; ++b_idx) {
      int b_row = B->rows[b_idx];
      b_value = B->values[b_idx];
      for (int a_idx = A->col_ptr[b_row]; a_idx < A->col_ptr[b_row + 1]; ++a_idx) {
        int a_row = A->rows[a_idx];
        accumulator[a_row] += A->values[a_idx] * b_value;
        present_elements[a_row] = 1;
      }
    }
    // write column into matrix C
    int c_pos = C->col_ptr[b_col];
    for (int c_row = 0; c_row < C->row_num; ++c_row) {
      if (present_elements[c_row] != 0) {
        C->rows[c_pos] = c_row;
        C->values[c_pos++] = accumulator[c_row];
      }
    }
  }

  return true;
}

bool SpgemmCSCComplexTBBSeq::post_processing() {
  internal_order_test();

  return true;
}

bool SpgemmCSCComplexTBBPar::pre_processing() {
  internal_order_test();

  A = reinterpret_cast<sparse_matrix *>(taskData->inputs[0]);
  B = reinterpret_cast<sparse_matrix *>(taskData->inputs[1]);
  C = reinterpret_cast<sparse_matrix *>(taskData->outputs[0]);
  return true;
}

bool SpgemmCSCComplexTBBPar::validation() {
  internal_order_test();

  int A_col_num = reinterpret_cast<sparse_matrix *>(taskData->inputs[0])->col_num;
  int B_row_num = reinterpret_cast<sparse_matrix *>(taskData->inputs[1])->row_num;
  // check that matrices are compatible for multiplication
  return (A_col_num == B_row_num);
}

bool SpgemmCSCComplexTBBPar::run() {
  internal_order_test();

  // symbolic stage
  C->row_num = A->row_num;
  C->col_num = B->col_num;
  C->col_ptr.resize(C->col_num + 1);
  C->col_ptr[0] = 0;
  constexpr int grain_size = 32;
  oneapi::tbb::parallel_for(tbb::blocked_range<int>(0, C->col_num, grain_size), [&](tbb::blocked_range<int> &r) {
    std::vector<int> present_elements(C->row_num);
    // std::cout << "GOT [" << r.begin() << "; " << r.end() << ")\n";
    for (int b_col = r.begin(); b_col < r.end(); ++b_col) {
      for (int c_row = 0; c_row < C->row_num; ++c_row) {
        present_elements[c_row] = 0;
      }
      for (int b_idx = B->col_ptr[b_col]; b_idx < B->col_ptr[b_col + 1]; ++b_idx) {
        int b_row = B->rows[b_idx];
        for (int a_idx = A->col_ptr[b_row]; a_idx < A->col_ptr[b_row + 1]; ++a_idx) {
          present_elements[A->rows[a_idx]] = 1;
        }
      }
      int col_nonzero_count = 0;
      for (int c_row = 0; c_row < C->row_num; ++c_row) {
        col_nonzero_count += present_elements[c_row];
      }
      C->col_ptr[b_col + 1] = col_nonzero_count;
    }
  });

  // allocate memory for matrix C
  for (int c_col = 0; c_col < C->col_num; ++c_col) {
    C->col_ptr[c_col + 1] += C->col_ptr[c_col];
  }
  int total_nonzeros = C->col_ptr[C->col_num];
  C->nonzeros = total_nonzeros;
  C->rows.resize(total_nonzeros);
  C->values.resize(total_nonzeros);

  // numeric stage
  oneapi::tbb::parallel_for(tbb::blocked_range<int>(0, C->col_num, grain_size), [&](tbb::blocked_range<int> &r) {
    std::complex<double> zero;
    std::complex<double> b_value;
    std::vector<std::complex<double>> accumulator(C->row_num);
    std::vector<int> present_elements(C->row_num);
    for (int b_col = r.begin(); b_col < r.end(); ++b_col) {
      for (int c_row = 0; c_row < C->row_num; ++c_row) {
        accumulator[c_row] = zero;
        present_elements[c_row] = 0;
      }
      // calculate column into accumulator
      for (int b_idx = B->col_ptr[b_col]; b_idx < B->col_ptr[b_col + 1]; ++b_idx) {
        int b_row = B->rows[b_idx];
        b_value = B->values[b_idx];
        for (int a_idx = A->col_ptr[b_row]; a_idx < A->col_ptr[b_row + 1]; ++a_idx) {
          int a_row = A->rows[a_idx];
          accumulator[a_row] += A->values[a_idx] * b_value;
          present_elements[a_row] = 1;
        }
      }
      // write column into matrix C
      int c_pos = C->col_ptr[b_col];
      for (int c_row = 0; c_row < C->row_num; ++c_row) {
        if (present_elements[c_row] != 0) {
          C->rows[c_pos] = c_row;
          C->values[c_pos++] = accumulator[c_row];
        }
      }
    }
  });

  return true;
}

bool SpgemmCSCComplexTBBPar::post_processing() {
  internal_order_test();

  return true;
}
