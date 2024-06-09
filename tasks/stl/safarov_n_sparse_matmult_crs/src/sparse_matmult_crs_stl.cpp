// Copyright 2024 Safarov Nurlan
#include "stl/safarov_n_sparse_matmult_crs/include/sparse_matmult_crs_stl.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <utility>
#include <vector>

SparseMatrixCRS::SparseMatrixCRS(int _numberOfColumns, int _numberOfRows, const std::vector<double>& _values,
                                 const std::vector<int>& _columnIndexes, const std::vector<int>& _pointers)
    : numberOfColumns(_numberOfColumns),
      numberOfRows(_numberOfRows),
      values(_values),
      columnIndexes(_columnIndexes),
      pointers(_pointers) {}

SparseMatrixCRS::SparseMatrixCRS(int _numberOfColumns, int _numberOfRows) {
  numberOfColumns = _numberOfColumns;
  numberOfRows = _numberOfRows;
}

SparseMatrixCRS::SparseMatrixCRS(std::vector<std::vector<double>> matrix) {
  int indexCounter = 0;
  numberOfRows = matrix.size();
  numberOfColumns = matrix[0].size();
  pointers.push_back(indexCounter);
  for (int r = 0; r < numberOfRows; r++) {
    for (int c = 0; c < numberOfColumns; c++) {
      if (matrix[r][c] != 0) {
        values.push_back(matrix[r][c]);
        indexCounter++;
        columnIndexes.push_back(c);
      }
    }
    pointers.push_back(indexCounter);
  }
}

SparseMatrixCRS sparseMatrixTransposeCRS(const SparseMatrixCRS& object) {
  SparseMatrixCRS matrix;
  std::vector<std::vector<int>> locCVec(object.numberOfColumns);
  std::vector<std::vector<double>> locVecVal(object.numberOfColumns);
  matrix.numberOfColumns = object.numberOfRows;
  int elementCounter = 0;
  matrix.numberOfRows = object.numberOfColumns;

  for (int r = 0; r < object.numberOfRows; r++) {
    for (int index = object.pointers[r]; index < object.pointers[r + 1]; index++) {
      int cIndex = object.columnIndexes[index];
      locCVec[cIndex].push_back(r);
      locVecVal[cIndex].push_back(object.values[index]);
    }
  }
  matrix.pointers.push_back(elementCounter);
  for (int c = 0; c < object.numberOfColumns; c++) {
    for (size_t ktmp = 0; ktmp < locCVec[c].size(); ktmp++) {
      matrix.columnIndexes.push_back(locCVec[c][ktmp]);
      matrix.values.push_back(locVecVal[c][ktmp]);
    }
    elementCounter += locCVec[c].size();
    matrix.pointers.push_back(elementCounter);
  }
  return matrix;
}

bool SparseMatrixCRS::operator==(const SparseMatrixCRS& matrix) const {
  return (values == matrix.values) && (numberOfColumns == matrix.numberOfColumns) &&
         (columnIndexes == matrix.columnIndexes) && (numberOfRows == matrix.numberOfRows) &&
         (pointers == matrix.pointers);
}

std::vector<std::vector<double>> fillTheMatrixWithZeros(int columns, int rows) {
  std::vector<std::vector<double>> result(rows);
  for (int m = 0; m < rows; m++) {
    for (int n = 0; n < columns; n++) {
      result[m].push_back(0);
    }
  }
  return result;
}

std::vector<std::vector<double>> multiplyMatrices(std::vector<std::vector<double>> A,
                                                  std::vector<std::vector<double>> B) {
  int p = B[0].size();
  int q = A.size();
  std::vector<std::vector<double>> resultMatrix = fillTheMatrixWithZeros(p, q);
  for (int rr = 0; rr < q; rr++) {
    for (int cc = 0; cc < p; cc++) {
      resultMatrix[rr][cc] = 0;
      for (size_t k = 0; k < A[0].size(); k++) {
        resultMatrix[rr][cc] += A[rr][k] * B[k][cc];
      }
    }
  }
  return resultMatrix;
}

std::vector<std::vector<double>> createRandomMatrix(int columns, int rows, double perc) {
  if (perc < 0 || perc > 1) {
    throw std::runtime_error("Wrong density. \n");
  }
  std::random_device mydev;
  std::vector<std::vector<double>> result = fillTheMatrixWithZeros(columns, rows);
  std::mt19937 gen(mydev());
  std::uniform_real_distribution<double> genP{0.0, 1.0};
  std::uniform_real_distribution<double> genVal{0.0, 25.0};
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < columns; c++) {
      if (genP(gen) <= perc) {
        result[r][c] = genVal(gen);
      }
    }
  }
  return result;
}

bool verifyCRSAttributes(const SparseMatrixCRS& object) {
  int nonZeroCount = object.values.size();
  auto check = size_t(nonZeroCount);

  if (object.pointers.size() != size_t(object.numberOfRows + 1)) {
    return false;
  }
  if (object.pointers[0] != 0) {
    return false;
  }
  if (object.values.size() != check || object.columnIndexes.size() != check ||
      object.pointers[object.numberOfRows] != nonZeroCount) {
    return false;
  }

  for (int i = 0; i < nonZeroCount; ++i) {
    if (object.columnIndexes[i] < 0 || object.columnIndexes[i] >= object.numberOfColumns) {
      return false;
    }
  }

  for (int j = 1; j <= object.numberOfRows; ++j) {
    if (object.pointers[j - 1] > object.pointers[j]) {
      return false;
    }
  }

  return true;
}

bool SparseMatrixMultiplicationCRS_STL::validation() {
  internal_order_test();

  X = reinterpret_cast<SparseMatrixCRS*>(taskData->inputs[0]);
  Y = reinterpret_cast<SparseMatrixCRS*>(taskData->inputs[1]);
  Z = reinterpret_cast<SparseMatrixCRS*>(taskData->outputs[0]);

  if (X == nullptr || Y == nullptr || Z == nullptr) {
    return false;
  }

  if (!verifyCRSAttributes(*X) || !verifyCRSAttributes(*Y)) {
    return false;
  }

  if (taskData->inputs.size() != 2 || taskData->outputs.size() != 1 || !taskData->inputs_count.empty() ||
      !taskData->outputs_count.empty()) {
    return false;
  }

  if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr || taskData->outputs[0] == nullptr) {
    return false;
  }

  if (X->numberOfColumns != Y->numberOfRows) {
    return false;
  }

  return true;
}

bool SparseMatrixMultiplicationCRS_STL::pre_processing() {
  internal_order_test();

  X = reinterpret_cast<SparseMatrixCRS*>(taskData->inputs[0]);
  Y = reinterpret_cast<SparseMatrixCRS*>(taskData->inputs[1]);
  Z = reinterpret_cast<SparseMatrixCRS*>(taskData->outputs[0]);

  *Y = sparseMatrixTransposeCRS(*Y);
  return true;
}

bool SparseMatrixMultiplicationCRS_STL::run() {
  internal_order_test();

  std::vector<int> finalColumnIndexes;
  std::vector<int> finalPointers;
  std::vector<double> finalValues;
  int resultRows = X->numberOfRows;
  std::vector<std::vector<int>> localColumnIndexes(X->numberOfRows);
  std::vector<std::vector<double>> localValues(X->numberOfRows);

  int resultColumnIndexes = Y->numberOfRows;  // After transposing matrix Y

  const int num_threads = 4;
  std::vector<std::thread> threads(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    threads[i] = std::thread([&, i]() {
      for (int rOne = i; rOne < X->numberOfRows; rOne += num_threads) {
        for (int rTwo = 0; rTwo < Y->numberOfRows; rTwo++) {
          int firstCurrentPointer = X->pointers[rOne];
          int secondCurrentPointer = Y->pointers[rTwo];
          int firstEndPointer = X->pointers[rOne + 1] - 1;
          int secondEndPointer = Y->pointers[rTwo + 1] - 1;
          double v = 0;

          while ((secondCurrentPointer <= secondEndPointer) && (firstCurrentPointer <= firstEndPointer)) {
            if (X->columnIndexes[firstCurrentPointer] <= Y->columnIndexes[secondCurrentPointer]) {
              if (X->columnIndexes[firstCurrentPointer] == Y->columnIndexes[secondCurrentPointer]) {
                v += X->values[firstCurrentPointer] * Y->values[secondCurrentPointer];
                secondCurrentPointer++;
                firstCurrentPointer++;
              } else {
                firstCurrentPointer++;
              }
            } else {
              secondCurrentPointer++;
            }
          }
          if (v != 0) {
            localValues[rOne].push_back(v);
            localColumnIndexes[rOne].push_back(rTwo);
          }
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  int elementCounter = 0;
  finalPointers.push_back(elementCounter);

  for (int indRow = 0; indRow < X->numberOfRows; indRow++) {
    elementCounter = elementCounter + localColumnIndexes[indRow].size();
    finalColumnIndexes.insert(finalColumnIndexes.end(), localColumnIndexes[indRow].begin(),
                              localColumnIndexes[indRow].end());
    finalValues.insert(finalValues.end(), localValues[indRow].begin(), localValues[indRow].end());
    finalPointers.push_back(elementCounter);
  }

  Z->numberOfColumns = resultColumnIndexes;
  Z->numberOfRows = resultRows;
  Z->values = finalValues;
  Z->columnIndexes = finalColumnIndexes;
  Z->pointers = finalPointers;

  return true;
}

bool SparseMatrixMultiplicationCRS_STL::post_processing() {
  internal_order_test();

  return true;
}