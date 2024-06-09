// Copyright 2024 Isaev Dmitriy
#include "seq/isaev_d_sparse_multe_double_crs/include/ops_seq.hpp"

#include <random>
#include <thread>

namespace IsaevSeq {

SparseMatrix getRandomMatrix(int _rows, int _columns, double chance, int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> rnd(-5.0, 5.0);
  std::bernoulli_distribution bd(chance);

  SparseMatrix randomMatrix;

  randomMatrix.rows = _rows;
  randomMatrix.columns = _columns;
  randomMatrix.row_pointers.assign(randomMatrix.rows + 1, 0);

  std::vector<std::vector<std::pair<int, double>>> temp(randomMatrix.rows);

  for (int i = 0; i < _rows; i++) {
    for (int j = 0; j < _columns; j++) {
      if (bd(gen)) {
        double v = rnd(gen);
        temp[i].emplace_back(j, v);
      }
    }
  }

  for (int i = 0; i < randomMatrix.rows; i++) {
    randomMatrix.row_pointers[i + 1] = randomMatrix.row_pointers[i];
    for (const auto &t : temp[i]) {
      randomMatrix.column_indices.push_back(t.first);
      randomMatrix.values.push_back(t.second);
      randomMatrix.row_pointers[i + 1]++;
    }
  }

  return randomMatrix;
}

bool SparseMultDoubleCRS::pre_processing() {
  internal_order_test();
  // Init value for input and output
  A = reinterpret_cast<SparseMatrix *>(taskData->inputs[0]);
  B = reinterpret_cast<SparseMatrix *>(taskData->inputs[1]);
  C = reinterpret_cast<SparseMatrix *>(taskData->outputs[0]);

  C->rows = A->rows;
  C->columns = B->columns;

  // Allocate memory for result
  // C->values.resize(C->rows * C->columns, 0.0);
  C->values.clear();
  C->column_indices.clear();
  // C->row_pointers.resize(C->rows + 1, 0);
  return true;
}

bool SparseMultDoubleCRS::validation() {
  internal_order_test();
  A = reinterpret_cast<SparseMatrix *>(taskData->inputs[0]);
  B = reinterpret_cast<SparseMatrix *>(taskData->inputs[1]);
  C = reinterpret_cast<SparseMatrix *>(taskData->outputs[0]);
  // Check count elements of output
  return B->rows == A->columns;
}

bool SparseMultDoubleCRS::run() {
  internal_order_test();

  std::vector<std::vector<double>> temp(C->rows, std::vector<double>(C->columns, 0.0));
  for (int i = 0; i < A->rows; ++i) {
    for (int k = A->row_pointers[i]; k < A->row_pointers[i + 1]; ++k) {
      int j = A->column_indices[k];
      for (int l = B->row_pointers[j]; l < B->row_pointers[j + 1]; ++l) {
        int m = B->column_indices[l];
        temp[i][m] += A->values[k] * B->values[l];
      }
    }
  }

  int nnz = 0;
  for (int i = 0; i < C->rows; ++i) {
    C->row_pointers.push_back(nnz);
    for (int j = 0; j < C->columns; ++j) {
      if (temp[i][j] != 0.0) {
        C->values.push_back(temp[i][j]);
        C->column_indices.push_back(j);
        nnz++;
      }
    }
    // C->row_pointers.push_back(nnz);
  }
  C->row_pointers.push_back(nnz);

  // std::vector<std::vector<double>> temp(C->rows, std::vector<double>(C->columns, 0.0));
  // for (int i = 0; i < A->rows; ++i) {
  //   for (int k = A->row_pointers[i]; k < A->row_pointers[i + 1]; ++k) {
  //     int j = A->column_indices[k];
  //     for (int l = B->row_pointers[j]; l < B->row_pointers[j + 1]; ++l) {
  //       int m = B->column_indices[l];
  //       temp[i][m] += A->values[k] * B->values[l];
  //     }
  //   }
  // }

  // int nnz = 0;
  // for (int i = 0; i < C->rows; ++i) {
  //   C->row_pointers.push_back(nnz);  // Add row pointer for row i

  //   for (int j = 0; j < C->columns; ++j) {
  //     if (temp[i][j] != 0.0) {
  //       C->values.push_back(temp[i][j]);  // Add non-zero value to values array
  //       C->column_indices.push_back(j);   // Add corresponding column index
  //       nnz++;
  //     }
  //   }
  // }
  // C->row_pointers.push_back(nnz);  // Add final row pointer

  return true;
}

bool SparseMultDoubleCRS::post_processing() {
  internal_order_test();

  return true;
}

}  // namespace IsaevSeq