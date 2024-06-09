// Copyright 2024 Sadikov Damir
#include "omp/sadikov_d_crs_mult/include/sadikov_d_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <complex>
#include <utility>
#include <vector>

bool check_CRS_properties(const matrix_CRS& A) {
  if (A.row_id.size() != size_t(A.n + 1)) return false;
  int nz = A.value.size();
  if (A.value.size() != size_t(nz) || A.col.size() != size_t(nz) || A.row_id[A.n] != nz) return false;
  if (A.row_id[0] != 0) return false;
  for (int i = 1; i <= A.n; i++) {
    if (A.row_id[i] < A.row_id[i - 1]) return false;
  }
  for (int i = 0; i < nz; i++) {
    if (A.col[i] < 0 || A.col[i] >= A.m) return false;
  }
  return true;
}

matrix_CRS transpose_CRS(const matrix_CRS& B) {
  matrix_CRS B_T;
  B_T.n = B.m;
  B_T.m = B.n;
  B_T.row_id.assign(B_T.n + 1, 0);
  std::vector<std::vector<std::pair<int, std::complex<double>>>> temp(B_T.n);
  for (int i = 0; i < B.n; i++) {
    for (int k = B.row_id[i]; k < B.row_id[i + 1]; k++) {
      int j = B.col[k];
      temp[j].emplace_back(i, B.value[k]);
    }
  }
  for (int i = 0; i < B_T.n; i++) {
    B_T.row_id[i + 1] = B_T.row_id[i];
    for (auto& j : temp[i]) {
      B_T.col.push_back(j.first);
      B_T.value.push_back(j.second);
      B_T.row_id[i + 1]++;
    }
  }
  return B_T;
}

bool CRSComplexMult_omp::validation() {
  internal_order_test();
  // check taskData vectors sizes
  if (taskData->inputs.size() != 2 || taskData->outputs.size() != 1 || !taskData->inputs_count.empty() ||
      !taskData->outputs_count.empty())
    return false;
  // check for nullptr
  if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr || taskData->outputs[0] == nullptr) return false;
  A = reinterpret_cast<matrix_CRS*>(taskData->inputs[0]);
  B = reinterpret_cast<matrix_CRS*>(taskData->inputs[1]);
  C = reinterpret_cast<matrix_CRS*>(taskData->outputs[0]);
  if (A == nullptr || B == nullptr || C == nullptr) return false;
  // check for CRS properties
  if (!check_CRS_properties(*A)) return false;
  if (!check_CRS_properties(*B)) return false;
  // check for matrices size
  return A->m == B->n;
}

bool CRSComplexMult_omp::pre_processing() {
  internal_order_test();
  // convert pointers
  A = reinterpret_cast<matrix_CRS*>(taskData->inputs[0]);
  B = reinterpret_cast<matrix_CRS*>(taskData->inputs[1]);
  C = reinterpret_cast<matrix_CRS*>(taskData->outputs[0]);
  // transpose B
  *B = transpose_CRS(*B);
  return true;
}

bool CRSComplexMult_omp::run() {
  internal_order_test();
  // multiply matrcies ans store result in C
  C->n = A->n;
  C->m = B->n;  // not m because B is transposed
  C->row_id.assign(C->n + 1, 0);
  std::vector<std::vector<std::pair<int, std::complex<double>>>> temp(C->n);
  omp_set_num_threads(4);
#pragma omp parallel for
  for (int i = 0; i < A->n; i++) {
    for (int j = 0; j < B->n; j++) {
      // C[i][j] = dot_product(A[i], B[j]);
      std::complex<double> T;
      for (int k_A = A->row_id[i]; k_A < A->row_id[i + 1]; k_A++) {
        for (int k_B = B->row_id[j]; k_B < B->row_id[j + 1]; k_B++) {
          if (A->col[k_A] == B->col[k_B]) {
            T += A->value[k_A] * B->value[k_B];
          }
        }
      }
      if (std::abs(T.imag()) > 1e-6 || std::abs(T.real()) > 1e-6) {
        temp[i].emplace_back(j, T);
      }
    }
  }
  for (int i = 0; i < C->n; i++) {
    C->row_id[i + 1] = C->row_id[i];
    for (auto& j : temp[i]) {
      C->col.push_back(j.first);
      C->value.push_back(j.second);
      C->row_id[i + 1]++;
    }
  }
  return true;
}

bool CRSComplexMult_omp::post_processing() {
  internal_order_test();
  return true;
}
