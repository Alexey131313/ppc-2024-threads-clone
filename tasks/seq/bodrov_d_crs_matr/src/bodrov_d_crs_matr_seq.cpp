// Copyright 2024 Bodrov Daniil
#include "seq/bodrov_d_crs_matr/include/bodrov_d_crs_matr_seq.hpp"

#include <algorithm>
#include <complex>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

SparseMatrix T(const SparseMatrix& M) {
  SparseMatrix temp_matrix{M.n_cols, M.n_rows, {}, {}, {}};

  std::vector<std::vector<std::pair<int, std::complex<double>>>> columns(M.n_cols);

  for (int i = 0; i < M.n_rows; ++i) {
    for (int k = M.pointer[i]; k < M.pointer[i + 1]; ++k) {
      int j = M.col_indexes[k];
      columns[j].emplace_back(i, M.non_zero_values[k]);
    }
  }

  for (int i = 0; i < M.n_cols; ++i) {
    temp_matrix.pointer.push_back(static_cast<int>(temp_matrix.non_zero_values.size()));
    for (const auto& item : columns[i]) {
      temp_matrix.col_indexes.push_back(item.first);
      temp_matrix.non_zero_values.push_back(item.second);
    }
  }
  temp_matrix.pointer.push_back(static_cast<int>(temp_matrix.non_zero_values.size()));

  return temp_matrix;
}

bool isValidPointer(const SparseMatrix& M) {
  if (M.pointer.size() != static_cast<size_t>(M.n_rows + 1)) return false;

  for (int i = 1; i <= M.n_rows; i++) {
    if (M.pointer[i] < M.pointer[i - 1]) return false;
  }

  return true;
}

bool isValidColumnIndexes(const SparseMatrix& M) {
  int non_zero_elems_count = M.non_zero_values.size();

  if (M.col_indexes.size() != static_cast<size_t>(non_zero_elems_count)) return false;

  for (int i = 0; i < non_zero_elems_count; i++) {
    if (M.col_indexes[i] < 0 || M.col_indexes[i] >= M.n_cols) return false;
  }

  return true;
}

bool IsCRS(const SparseMatrix& M) {
  if (!isValidPointer(M)) return false;

  if (M.pointer[0] != 0 || M.pointer[M.n_rows] != static_cast<int>(M.non_zero_values.size())) return false;

  if (!isValidColumnIndexes(M)) return false;

  return true;
}

bool SparseMatrixSolver::pre_processing() {
  internal_order_test();

  *B_M = T(*B_M);

  return true;
}

bool SparseMatrixSolver::validation() {
  internal_order_test();

  if (taskData->inputs.size() != 2 || taskData->outputs.size() != 1 || !taskData->inputs_count.empty() ||
      !taskData->outputs_count.empty())
    return false;

  A_M = reinterpret_cast<SparseMatrix*>(taskData->inputs[0]);
  B_M = reinterpret_cast<SparseMatrix*>(taskData->inputs[1]);
  Result = reinterpret_cast<SparseMatrix*>(taskData->outputs[0]);

  if (A_M == nullptr || B_M == nullptr || Result == nullptr) return false;

  if (!IsCRS(*A_M) || !IsCRS(*B_M)) return false;

  if (A_M->n_cols != B_M->n_rows) return false;

  return true;
}

std::complex<double> computeDotProduct(const SparseMatrix& A_M, const SparseMatrix& B_M, int row_A, int row_B) {
  std::complex<double> result;
  int k_A = A_M.pointer[row_A];
  int k_B = B_M.pointer[row_B];

  while (k_A < A_M.pointer[row_A + 1] && k_B < B_M.pointer[row_B + 1]) {
    int col_A = A_M.col_indexes[k_A];
    int col_B = B_M.col_indexes[k_B];

    if (col_A == col_B) {
      result += A_M.non_zero_values[k_A] * B_M.non_zero_values[k_B];
      ++k_A;
      ++k_B;
    } else if (col_A < col_B) {
      ++k_A;
    } else {
      ++k_B;
    }
  }

  return result;
}

bool isNonZero(const std::complex<double>& value) { return std::norm(value) > 1e-6; }

bool SparseMatrixSolver::run() {
  internal_order_test();

  Result->n_rows = A_M->n_rows;
  Result->n_cols = B_M->n_rows;
  Result->pointer.assign(Result->n_rows + 1, 0);

  std::vector<std::vector<std::pair<int, std::complex<double>>>> temp(Result->n_rows);

  for (int i = 0; i < Result->n_rows; ++i) {
    for (int j = 0; j < B_M->n_rows; ++j) {
      std::complex<double> product = computeDotProduct(*A_M, *B_M, i, j);
      if (isNonZero(product)) {
        temp[i].emplace_back(j, product);
      }
    }
  }

  for (int i = 0; i < Result->n_rows; ++i) {
    Result->pointer[i + 1] = Result->pointer[i];
    for (const auto& pair : temp[i]) {
      Result->col_indexes.push_back(pair.first);
      Result->non_zero_values.push_back(pair.second);
      Result->pointer[i + 1]++;
    }
  }

  return true;
}

bool SparseMatrixSolver::post_processing() {
  internal_order_test();
  return true;
}