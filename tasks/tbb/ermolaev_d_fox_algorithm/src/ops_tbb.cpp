#include "tbb/ermolaev_d_fox_algorithm/include/ops_tbb.hpp"

#include <tbb/tbb.h>

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

bool FoxAlgorithm::validation() {
  internal_order_test();
  return !taskData->inputs_count.empty() && taskData->inputs_count.size() == 2 &&
         taskData->inputs_count[0] == taskData->inputs_count[1] && taskData->outputs_count.size() == 1 &&
         taskData->outputs_count[0] == taskData->inputs_count[0] && taskData->inputs[0] != nullptr &&
         taskData->inputs[1] != nullptr && taskData->outputs[0] != nullptr;
}

bool FoxAlgorithm::pre_processing() {
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

bool FoxAlgorithm::run() {
  internal_order_test();
  try {
    int n = static_cast<int>(data_size);
    int blockSize = calculateBlockSize(n);
    int numBlocks = n / blockSize;

    std::vector<std::vector<double>> result(data_size, std::vector<double>(data_size, 0.0));

    tbb::parallel_for(0, numBlocks, [&](int stage) {
      std::vector<std::vector<double>> local_matrix_C(data_size, std::vector<double>(data_size, 0.0));

      for (int i = 0; i < numBlocks; ++i) {
        for (int j = 0; j < numBlocks; ++j) {
          int k = (i + stage) % numBlocks;
          std::vector<std::vector<double>> block_A(blockSize, std::vector<double>(blockSize));
          std::vector<std::vector<double>> block_B(blockSize, std::vector<double>(blockSize));
          for (int row = 0; row < blockSize; ++row) {
            for (int col = 0; col < blockSize; ++col) {
              block_A[row][col] = matrix_A[((i * blockSize) + row) * n + (k * blockSize) + col];
              block_B[row][col] = matrix_B[((k * blockSize) + row) * n + (j * blockSize) + col];
            }
          }
          std::vector<std::vector<double>> block_C = multiply_block(block_A, block_B);
          for (int row = 0; row < blockSize; ++row) {
            for (int col = 0; col < blockSize; ++col) {
              local_matrix_C[i * blockSize + row][j * blockSize + col] += block_C[row][col];
            }
          }
        }
      }

      static tbb::mutex mutex;
      tbb::mutex::scoped_lock lock(mutex);
      for (size_t i = 0; i < data_size; ++i) {
        for (size_t j = 0; j < data_size; ++j) {
          matrix_C[i * data_size + j] += local_matrix_C[i][j];
        }
      }
    });
  } catch (...) {
    return false;
  }
  return true;
  return true;
}

bool FoxAlgorithm::post_processing() {
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