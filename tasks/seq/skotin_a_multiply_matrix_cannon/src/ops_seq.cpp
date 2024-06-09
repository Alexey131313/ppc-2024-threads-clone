// Copyright 2024 Skotin Alexander
#include "seq/skotin_a_multiply_matrix_cannon/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

bool MatrixMultiplicationTask::pre_processing() {
  size_t totalElementsPerMatrix = taskData->inputs_count[0] / sizeof(double);
  size_t matrixSize = std::sqrt(totalElementsPerMatrix);

  if (matrixSize * matrixSize != totalElementsPerMatrix) {
    std::cerr << "Incorrect matrix size." << std::endl;
    return false;
  }

  std::vector<double> matrixAData(matrixSize * matrixSize);
  memcpy(matrixAData.data(), taskData->inputs[0], taskData->inputs_count[0]);

  if (!loadMatrix(matrixAData, matrixA, matrixSize)) {
    std::cerr << "Failed to load matrix A." << std::endl;
    return false;
  }

  std::vector<double> matrixBData(matrixSize * matrixSize);
  memcpy(matrixBData.data(), taskData->inputs[1], taskData->inputs_count[1]);

  if (!loadMatrix(matrixBData, matrixB, matrixSize)) {
    std::cerr << "Failed to load matrix B." << std::endl;
    return false;
  }

  resultMatrix.resize(matrixSize, std::vector<double>(matrixSize, 0));
  return true;
}

bool MatrixMultiplicationTask::validation() {
  if (matrixA.empty() || matrixB.empty()) return false;
  return matrixA[0].size() == matrixB.size();
}

bool MatrixMultiplicationTask::run() {
  size_t blockSize = std::sqrt(matrixA.size());
  size_t n = matrixA.size();
  resultMatrix.resize(n, std::vector<double>(n, 0.0));

  for (size_t blockRow = 0; blockRow < n; blockRow += blockSize) {
    for (size_t blockCol = 0; blockCol < n; blockCol += blockSize) {
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = blockCol; j < std::min(blockCol + blockSize, n); ++j) {
          double sum = 0.0;
          for (size_t k = blockRow; k < std::min(blockRow + blockSize, n); ++k) {
            sum += matrixA[i][k] * matrixB[k][j];
          }
          resultMatrix[i][j] += sum;
        }
      }
    }
  }

  return true;
}

bool MatrixMultiplicationTask::post_processing() { return saveResult(); }

bool MatrixMultiplicationTask::loadMatrix(const std::vector<double>& inputData,
                                          std::vector<std::vector<double>>& matrix, size_t size) {
  if (inputData.size() != size * size) {
    std::cerr << "Input data size does not match expected matrix size." << std::endl;
    return false;
  }
  if (inputData.size() != size * size) return false;
  matrix.resize(size, std::vector<double>(size, 0));
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[i][j] = inputData[i * size + j];
    }
  }
  return true;
}

bool MatrixMultiplicationTask::saveResult() {
  size_t totalBytes = resultMatrix.size() * resultMatrix[0].size() * sizeof(double);

  std::vector<uint8_t> outputData(totalBytes);

  for (size_t i = 0; i < resultMatrix.size(); ++i) {
    for (size_t j = 0; j < resultMatrix[i].size(); ++j) {
      double value = resultMatrix[i][j];
      memcpy(&outputData[(i * resultMatrix[i].size() + j) * sizeof(double)], &value, sizeof(double));
    }
  }

  std::copy(outputData.begin(), outputData.end(), taskData->outputs[0]);

  taskData->outputs_count[0] = totalBytes;
  return true;
}
