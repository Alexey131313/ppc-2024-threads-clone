// Copyright 2023 Belan Vadim
#include "seq/belan_vadim_mat_fox_alg/include/ops_seq.hpp"

#include <cmath>
#include <cstring>

bool MatrixMulFox::pre_processing() {
  int n = reinterpret_cast<int*>(taskData->inputs[0])[0];
  int block_size = reinterpret_cast<int*>(taskData->inputs[0])[1];

  matrix_a_.resize(n * n);
  matrix_b_.resize(n * n);
  matrix_c_.resize(n * n);
  std::memcpy(matrix_a_.data(), taskData->inputs[1], n * n * sizeof(double));
  std::memcpy(matrix_b_.data(), taskData->inputs[2], n * n * sizeof(double));

  if (n <= 0 || block_size <= 0 || n % block_size != 0) {
    return false;
  }

  std::fill(matrix_c_.begin(), matrix_c_.end(), 0.0);

  n_ = n;
  block_size_ = block_size;

  return true;
}

bool MatrixMulFox::validation() { return true; }

bool MatrixMulFox::run() {
  fox_algorithm();
  return true;
}

bool MatrixMulFox::post_processing() {
  std::memcpy(taskData->outputs[0], matrix_c_.data(), n_ * n_ * sizeof(double));
  return true;
}

void MatrixMulFox::fox_algorithm() {
  int num_blocks = n_ / block_size_;

  for (int k = 0; k < num_blocks; k++) {
    for (int i = 0; i < num_blocks; i++) {
      for (int j = 0; j < num_blocks; j++) {
        for (int ii = 0; ii < block_size_; ii++) {
          for (int jj = 0; jj < block_size_; jj++) {
            for (int kk = 0; kk < block_size_; kk++) {
              int row_a = (i + ii) % num_blocks;
              int col_a = (k + kk) % num_blocks;
              int row_b = (k + kk) % num_blocks;
              int col_b = (j + jj) % num_blocks;
              int row_c = (i + ii) % num_blocks;
              int col_c = (j + jj) % num_blocks;
              matrix_c_[(row_c * block_size_ + ii) * n_ + col_c * block_size_ + jj] +=
                  matrix_a_[(row_a * block_size_ + ii) * n_ + col_a * block_size_ + kk] *
                  matrix_b_[(row_b * block_size_ + kk) * n_ + col_b * block_size_ + jj];
            }
          }
        }
      }
    }
  }
}
