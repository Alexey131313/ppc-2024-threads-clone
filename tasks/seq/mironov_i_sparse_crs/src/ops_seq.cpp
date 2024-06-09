// Copyright 2024 Nesterov Alexander
#include "seq/mironov_i_sparse_crs/include/ops_seq.hpp"

#include <cstring>
#include <random>
#include <vector>
const double EPS = 1e-6;

MatrixCRS::MatrixCRS(int n, int nz) : N(n), NZ(nz) {
  Value.reserve(nz);
  Col.reserve(nz);
  RowIndex.reserve(n + 1);
}

MatrixCRS::MatrixCRS(const double* matrix, int n, int m, bool transpose) {
  int index;
  if (transpose) {
    N = m;
    m = n;
  } else {
    N = n;
  }
  RowIndex.reserve(N + 1);
  NZ = 0;
  for (int i = 0; i < N; i++) {
    RowIndex[i] = NZ;
    for (int j = 0; j < m; j++) {
      index = transpose ? i + N * j : i * m + j;
      if (std::fabs(matrix[index]) >= EPS) {
        Value.push_back(matrix[index]);
        Col.push_back(j);
        NZ++;
      }
    }
  }
  RowIndex[N] = NZ;
}

MatrixCRS Multiplicate2(const MatrixCRS& A, const MatrixCRS& BT, int m) {
  int N = A.N;
  int nz_expected = A.NZ * BT.NZ / N + 1;
  std::vector<int> columns;
  std::vector<double> values;
  std::vector<int> row_index(N + 1);
  values.reserve(nz_expected);
  columns.reserve(nz_expected);
  std::vector<int> temp(m);
  int nz = 0;

  for (int i = 0; i < N; i++) {
    row_index[i] = nz;

    memset(temp.data(), 0, m * sizeof(*temp.data()));
    int ind1 = A.RowIndex[i];
    int ind2 = A.RowIndex[i + 1];
    for (int k = ind1; k < ind2; k++) {
      temp[A.Col[k]] = k + 1;
    }

    for (int j = 0; j < BT.N; j++) {
      double sum = 0;

      int ind3 = BT.RowIndex[j];
      int ind4 = BT.RowIndex[j + 1];
      for (int k = ind3; k < ind4; k++) {
        int aind = temp[BT.Col[k]];
        if (aind != 0) {
          sum += A.Value[aind - 1] * BT.Value[k];
        }
      }

      if (std::fabs(sum) > EPS) {
        columns.push_back(j);
        values.push_back(sum);
        nz++;
      }
    }
  }
  row_index[N] = nz;

  MatrixCRS C(N, nz);
  std::move(values.begin(), values.end(), C.Value.begin());
  std::move(columns.begin(), columns.end(), C.Col.begin());
  std::move(row_index.begin(), row_index.end(), C.RowIndex.begin());
  return C;
}

bool MironovISequential::pre_processing() {
  internal_order_test();
  M = taskData->inputs_count[1];
  A = MatrixCRS(reinterpret_cast<double*>(taskData->inputs[0]), taskData->inputs_count[0], M, false);
  BT = MatrixCRS(reinterpret_cast<double*>(taskData->inputs[1]), taskData->inputs_count[2], taskData->inputs_count[3],
                 true);
  c_out = reinterpret_cast<double*>(taskData->outputs[0]);
  return true;
}

bool MironovISequential::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs[0] != nullptr) && (taskData->inputs_count[0] != 0u) && (taskData->inputs[1] != nullptr) &&
         (taskData->inputs_count[1] != 0u) && (taskData->outputs[0] != nullptr) && (taskData->outputs_count[0] != 0u);
}

void PrintCRSMatrix(const MatrixCRS& matrix, int m) {
  for (int i = 0; i < matrix.N; i++) {
    int j = 0;
    for (int c_j = matrix.RowIndex[i]; c_j < matrix.RowIndex[i + 1]; c_j++) {
      while (j < matrix.Col[c_j]) {
        std::cout << "0 ";
        j++;
      }
      std::cout << matrix.Value[c_j] << " ";
      j++;
    }
    while (j < m) {
      std::cout << "0 ";
      j++;
    }
    std::cout << std::endl;
  }
}

bool MironovISequential::run() {
  internal_order_test();
  C = Multiplicate2(A, BT, M);
  return true;
}

bool MironovISequential::post_processing() {
  internal_order_test();
  int m = BT.N;
  int j;
  int c_j;

  for (int i = 0; i < C.N; i++) {
    j = 0;
    for (c_j = C.RowIndex[i]; c_j < C.RowIndex[i + 1]; c_j++) {
      while (j < C.Col[c_j]) {
        c_out[i * m + j] = 0;
        j++;
      }
      c_out[i * m + j] = C.Value[c_j];
      j++;
    }
    while (j < m) {
      c_out[i * m + j] = 0;
      j++;
    }
  }

  return true;
}

void MironovISequential::genrateSparseMatrix(double* matrix, int sz, double ro) {
  int nz = sz * ro;
  std::uniform_int_distribution<int> distribution(0, sz - 1);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> rand_r(-10.0, 10.0);
  for (int i = 0; i < nz; i++) {
    matrix[distribution(gen)] = rand_r(gen);
  }
}

void MironovISequential::genrateEdMatrix(double* matrix, int n) {
  for (int i = 0; i < n; i++) {
    matrix[i * n + i] = 1.0;
  }
}
