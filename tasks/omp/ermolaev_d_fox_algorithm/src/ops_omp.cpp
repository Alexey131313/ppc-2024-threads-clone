#include "omp/ermolaev_d_fox_algorithm/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>

using namespace std::chrono_literals;
namespace {
std::vector<std::vector<double>> multiply_block(const std::vector<std::vector<double>>& block_A,
                                                const std::vector<std::vector<double>>& block_B) {
  size_t blockSize = block_A.size();
  std::vector<std::vector<double>> block_C(blockSize, std::vector<double>(blockSize, 0.0));
  for (size_t i = 0; i < blockSize; ++i) {
    for (size_t j = 0; j < blockSize; ++j) {
      for (size_t k = 0; k < blockSize; ++k) {
        block_C[i][j] += block_A[i][k] * block_B[k][j];
      }
    }
  }
  return block_C;
}

int calculateBlockSize(int n, int cacheSize = 1024, int elementSize = 8) {
  int b = std::sqrt((cacheSize) / (elementSize * 2));

  while (n % b != 0) {
    b--;
  }
  return b;
}
}  // namespace

bool FoxAlgorithmOMP::validation() {
  internal_order_test();
  return !taskData->inputs_count.empty() && taskData->inputs_count.size() == 2 &&
         taskData->inputs_count[0] == taskData->inputs_count[1] && taskData->outputs_count.size() == 1 &&
         taskData->outputs_count[0] == taskData->inputs_count[0] && taskData->inputs[0] != nullptr &&
         taskData->inputs[1] != nullptr && taskData->outputs[0] != nullptr;
}

bool FoxAlgorithmOMP::pre_processing() {
  try {
    internal_order_test();
    data_size = static_cast<size_t>(sqrt(taskData->inputs_count[0]));
    if (data_size * data_size != taskData->inputs_count[0]) {
      std::cerr << "Data size error: input count does not match expected size of square matrix." << std::endl;
      return false;
    }
    matrix_A = reinterpret_cast<double*>(taskData->inputs[0]);
    matrix_B = reinterpret_cast<double*>(taskData->inputs[1]);
    matrix_C = reinterpret_cast<double*>(taskData->outputs[0]);
    for (size_t i = 0; i < data_size; ++i) {
      for (size_t j = 0; j < data_size; ++j) {
        matrix_C[i * data_size + j] = 0;
      }
    }
  } catch (...) {
    return false;
  }
  return matrix_A != nullptr && matrix_B != nullptr;
}

bool FoxAlgorithmOMP::run() {
  internal_order_test();
  double start = omp_get_wtime();
  try {
    int n = static_cast<int>(data_size);
    int blockSize = calculateBlockSize(n);
    int numBlocks = n / blockSize;

#pragma omp parallel
    {
      std::vector<std::vector<double>> block_A;
      std::vector<std::vector<double>> block_B;
      std::vector<std::vector<double>> block_C;
#pragma omp for nowait
      for (int stage = 0; stage < numBlocks; ++stage) {
        for (int i = 0; i < numBlocks; ++i) {
          for (int j = 0; j < numBlocks; ++j) {
            int k = (i + stage) % numBlocks;
            block_A.assign(blockSize, std::vector<double>(blockSize));
            block_B.assign(blockSize, std::vector<double>(blockSize));
            block_C.assign(blockSize, std::vector<double>(blockSize, 0.0));
            for (int row = 0; row < blockSize; ++row) {
              for (int col = 0; col < blockSize; ++col) {
                block_A[row][col] = matrix_A[((i * blockSize) + row) * n + (k * blockSize) + col];
                block_B[row][col] = matrix_B[((k * blockSize) + row) * n + (j * blockSize) + col];
              }
            }
            block_C = multiply_block(block_A, block_B);
#pragma omp critical
            for (int row = 0; row < blockSize; ++row) {
              for (int col = 0; col < blockSize; ++col) {
                matrix_C[((i * blockSize) + row) * n + (j * blockSize) + col] += block_C[row][col];
              }
            }
          }
        }
      }
    }
    double finish = omp_get_wtime();
    std::cout << "How measure time in OpenMP: " << finish - start << std::endl;
  } catch (...) {
    return false;
  }
  return true;
}

bool FoxAlgorithmOMP::post_processing() {
  internal_order_test();
  try {
    for (size_t i = 0; i < data_size; ++i) {
      for (size_t j = 0; j < data_size; ++j) {
        reinterpret_cast<double*>(taskData->outputs[0])[i * data_size + j] = matrix_C[i * data_size + j];
      }
    }
  } catch (...) {
    return false;
  }
  return true;
}