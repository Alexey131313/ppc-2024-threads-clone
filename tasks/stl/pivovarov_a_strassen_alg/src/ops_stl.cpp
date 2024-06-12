// Copyright 2024 Pivovarov Alexey
#include "stl/pivovarov_a_strassen_alg/include/ops_stl.hpp"

#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <future>
#include <algorithm>

namespace pivovarov_a_stl {

size_t log2(size_t n) {
  size_t res = 0;
  while (n != 0) {
    n >>= 1;
    res++;
  }
  return res;
}

size_t getNewSize(const std::vector<double>& a) {
  size_t n = std::sqrt(a.size());
  return static_cast<size_t>(std::pow(2, std::ceil(std::log2(n))));
}

std::vector<double> addSquareMatrix(const std::vector<double>& a, int n) {
  std::vector<double> res(n * n, 0.0);
  int size = std::sqrt(a.size());

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      res[i * n + j] = a[i * size + j];
    }
  }
  return res;
}

std::vector<double> multiplyMatrix(const std::vector<double>& A, const std::vector<double>& B, int n) {
  std::vector<double> C(n * n, 0.0);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
  return C;
}

std::vector<double> addMatrix(const std::vector<double>& A, const std::vector<double>& B, int n) {
  std::vector<double> C(n * n);
  for (int i = 0; i < n * n; ++i) {
    C[i] = A[i] + B[i];
  }
  return C;
}

std::vector<double> subMatrix(const std::vector<double>& A, const std::vector<double>& B, int n) {
  std::vector<double> C(n * n);
  for (int i = 0; i < n * n; ++i) {
    C[i] = A[i] - B[i];
  }
  return C;
}

void splitMatrix(const std::vector<double>& mSplit, std::vector<double>& a11, std::vector<double>& a12,
                 std::vector<double>& a21, std::vector<double>& a22, int halfSize) {
  for (int i = 0; i < halfSize; i++) {
    for (int j = 0; j < halfSize; j++) {
      a11[i * halfSize + j] = mSplit[i * halfSize * 2 + j];
      a12[i * halfSize + j] = mSplit[i * halfSize * 2 + j + halfSize];
      a21[i * halfSize + j] = mSplit[(i + halfSize) * halfSize * 2 + j];
      a22[i * halfSize + j] = mSplit[(i + halfSize) * halfSize * 2 + j + halfSize];
    }
  }
}

std::vector<double> mergeMatrix(const std::vector<double>& a11, const std::vector<double>& a12,
                                const std::vector<double>& a21, const std::vector<double>& a22, int halfSize) {
  int n = halfSize * 2;
  std::vector<double> res(n * n, 0.0);

  for (int i = 0; i < halfSize; i++) {
    for (int j = 0; j < halfSize; j++) {
      res[i * n + j] = a11[i * halfSize + j];
      res[i * n + j + halfSize] = a12[i * halfSize + j];
      res[(i + halfSize) * n + j] = a21[i * halfSize + j];
      res[(i + halfSize) * n + j + halfSize] = a22[i * halfSize + j];
    }
  }
  return res;
}

void strassenTask(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n,
                  int threadsCount, std::mutex& mutex);

std::vector<double> strassenMatrixMult(const std::vector<double>& A, const std::vector<double>& B, int n) {
  int threadsCount = std::thread::hardware_concurrency();
  if (threadsCount == 0) threadsCount = 1;

  std::vector<double> C(n * n, 0.0);
  std::mutex mutex;
  strassenTask(A, B, C, n, threadsCount, mutex);
  return C;
}

void strassenTask(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n,
                  int threadsCount, std::mutex& mutex) {
  const int BASE_SIZE = 4;

  if (n <= BASE_SIZE) {
    C = multiplyMatrix(A, B, n);
    return;
  }

  int halfSize = n / 2;
  int newSizeSquare = halfSize * halfSize;

  std::vector<double> A11(newSizeSquare);
  std::vector<double> A12(newSizeSquare);
  std::vector<double> A21(newSizeSquare);
  std::vector<double> A22(newSizeSquare);

  std::vector<double> B11(newSizeSquare);
  std::vector<double> B12(newSizeSquare);
  std::vector<double> B21(newSizeSquare);
  std::vector<double> B22(newSizeSquare);

  splitMatrix(A, A11, A12, A21, A22, halfSize);
  splitMatrix(B, B11, B12, B21, B22, halfSize);

  std::vector<double> P1(newSizeSquare), P2(newSizeSquare), P3(newSizeSquare), P4(newSizeSquare), P5(newSizeSquare),
      P6(newSizeSquare), P7(newSizeSquare);

  auto computeP = [&](std::vector<double>& P, const std::vector<double>& A, const std::vector<double>& B,
                      int halfSize) {
    strassenTask(A, B, P, halfSize, threadsCount, mutex);
  };

  std::vector<std::future<void>> futures;
  if (threadsCount > 1) {
    futures.push_back(std::async(std::launch::async, computeP, std::ref(P1), addMatrix(A11, A22, halfSize),
                                 addMatrix(B11, B22, halfSize), halfSize));
    futures.push_back(std::async(std::launch::async, computeP, std::ref(P2), addMatrix(A21, A22, halfSize), B11,
                                 halfSize));
    futures.push_back(std::async(std::launch::async, computeP, std::ref(P3), A11, subMatrix(B12, B22, halfSize),
                                 halfSize));
    futures.push_back(std::async(std::launch::async, computeP, std::ref(P4), A22, subMatrix(B21, B11, halfSize),
                                 halfSize));
    futures.push_back(std::async(std::launch::async, computeP, std::ref(P5), addMatrix(A11, A12, halfSize), B22,
                                 halfSize));
    futures.push_back(std::async(std::launch::async, computeP, std::ref(P6), subMatrix(A21, A11, halfSize),
                                 addMatrix(B11, B12, halfSize), halfSize));
    futures.push_back(std::async(std::launch::async, computeP, std::ref(P7), subMatrix(A12, A22, halfSize),
                                 addMatrix(B21, B22, halfSize), halfSize));

    for (auto& future : futures) {
      future.get();
    }
  } else {
    computeP(P1, addMatrix(A11, A22, halfSize), addMatrix(B11, B22, halfSize), halfSize);
    computeP(P2, addMatrix(A21, A22, halfSize), B11, halfSize);
    computeP(P3, A11, subMatrix(B12, B22, halfSize), halfSize);
    computeP(P4, A22, subMatrix(B21, B11, halfSize), halfSize);
    computeP(P5, addMatrix(A11, A12, halfSize), B22, halfSize);
    computeP(P6, subMatrix(A21, A11, halfSize), addMatrix(B11, B12, halfSize), halfSize);
    computeP(P7, subMatrix(A12, A22, halfSize), addMatrix(B21, B22, halfSize), halfSize);
  }

  std::vector<double> C11 = addMatrix(subMatrix(addMatrix(P1, P4, halfSize), P5, halfSize), P7, halfSize);
  std::vector<double> C12 = addMatrix(P3, P5, halfSize);
  std::vector<double> C21 = addMatrix(P2, P4, halfSize);
  std::vector<double> C22 = addMatrix(subMatrix(addMatrix(P1, P3, halfSize), P2, halfSize), P6, halfSize);

  std::vector<double> C_result = mergeMatrix(C11, C12, C21, C22, halfSize);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i * n + j] = C_result[i * n + j];
    }
  }
}

bool TestTaskSTLParallelPivovarovStrassen::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->inputs_count[1] &&
         taskData->inputs_count[0] == taskData->outputs_count[0] &&
         taskData->inputs_count[1] == taskData->outputs_count[0];
}

bool TestTaskSTLParallelPivovarovStrassen::pre_processing() {
  internal_order_test();

  A = std::vector<double>(taskData->inputs_count[0]);
  B = std::vector<double>(taskData->inputs_count[1]);

  n = *reinterpret_cast<int*>(taskData->inputs[2]);
  m = *reinterpret_cast<int*>(taskData->inputs[3]);

  auto* tmp_ptr_A = reinterpret_cast<double*>(taskData->inputs[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    A[i] = tmp_ptr_A[i];
  }

  auto* tmp_ptr_B = reinterpret_cast<double*>(taskData->inputs[1]);
  for (size_t i = 0; i < taskData->inputs_count[1]; i++) {
    B[i] = tmp_ptr_B[i];
  }
  return true;
}

bool TestTaskSTLParallelPivovarovStrassen::run() {
  internal_order_test();
  result = strassenMatrixMult(A, B, n);
  return true;
}

bool TestTaskSTLParallelPivovarovStrassen::post_processing() {
  internal_order_test();
  std::copy(result.begin(), result.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  return true;
}
}  // namespace pivovarov_a_stl