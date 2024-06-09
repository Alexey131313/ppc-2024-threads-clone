// Copyright 2024 Pivovarov Alexey
#include "stl/pivovarov_a_strassen_alg/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include <thread>
#include <future>

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

  if (n == 0) {
    return std::vector<double>();
  }

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
                 std::vector<double>& a21, std::vector<double>& a22) {
  int n = std::sqrt(mSplit.size()) / 2;

  for (int i = 0; i < n; i++) {
    std::copy(mSplit.begin() + (2 * i) * n, mSplit.begin() + (2 * i + 1) * n, a11.begin() + i * n);
    std::copy(mSplit.begin() + (2 * i + 1) * n, mSplit.begin() + (2 * i + 2) * n, a12.begin() + i * n);
    std::copy(mSplit.begin() + ((n * n * 2) + (2 * i * n)), mSplit.begin() + ((n * n * 2) + ((2 * i + 1) * n)),
              a21.begin() + i * n);
    std::copy(mSplit.begin() + ((n * n * 2) + ((2 * i + 1) * n)), mSplit.begin() + ((n * n * 2) + ((2 * i + 2) * n)),
              a22.begin() + i * n);
  }
}

std::vector<double> mergeMatrix(std::vector<double> a11, std::vector<double> a12, std::vector<double> a21,
                                std::vector<double> a22) {
  int n = a11.size();
  std::vector<double> res(4 * n, 0.0);
  n = std::sqrt(n);

  for (int i = 0; i < n; i++) {
    std::copy(a11.begin() + i * n, a11.begin() + (i + 1) * n, res.begin() + (2 * i * n));
    std::copy(a12.begin() + i * n, a12.begin() + (i + 1) * n, res.begin() + (2 * i + 1) * n);
    std::copy(a21.begin() + i * n, a21.begin() + (i + 1) * n, res.begin() + ((n * n * 2) + (2 * i * n)));
    std::copy(a22.begin() + i * n, a22.begin() + (i + 1) * n, res.begin() + ((n * n * 2) + ((2 * i + 1) * n)));
  }
  return res;
}

std::vector<double> strassenMatrixMult(const std::vector<double>& A, const std::vector<double>& B, int n) {
  std::vector<double> C(n * n, 0.0);

  if (n == 0) {
    return std::vector<double>();
  }

  int newSize = getNewSize(A);
  std::vector<double> newA = addSquareMatrix(A, newSize);
  std::vector<double> newB = addSquareMatrix(B, newSize);

  if (newSize == 1) {
    C[0] = newA[0] * newB[0];
  } else {
    int halfSize = newSize / 2;
    int newSizeSquare = halfSize * halfSize;

    std::vector<double> A11(newSizeSquare);
    std::vector<double> A12(newSizeSquare);
    std::vector<double> A21(newSizeSquare);
    std::vector<double> A22(newSizeSquare);

    std::vector<double> B11(newSizeSquare);
    std::vector<double> B12(newSizeSquare);
    std::vector<double> B21(newSizeSquare);
    std::vector<double> B22(newSizeSquare);

    splitMatrix(newA, A11, A12, A21, A22);
    splitMatrix(newB, B11, B12, B21, B22);

    std::vector<double> P1;
    std::vector<double> P2;
    std::vector<double> P3;
    std::vector<double> P4;
    std::vector<double> P5;
    std::vector<double> P6;
    std::vector<double> P7;

    std::future<std::vector<double>> f1 = std::async(std::launch::async, [&]() {
      return strassenMatrixMult(addMatrix(A11, A22, halfSize), addMatrix(B11, B22, halfSize), halfSize);
    });

    std::future<std::vector<double>> f2 = std::async(std::launch::async, [&]() {
      return strassenMatrixMult(addMatrix(A21, A22, halfSize), B11, halfSize);
    });

    std::future<std::vector<double>> f3 = std::async(std::launch::async, [&]() {
      return strassenMatrixMult(A11, subMatrix(B12, B22, halfSize), halfSize);
    });

    std::future<std::vector<double>> f4 = std::async(std::launch::async, [&]() {
      return strassenMatrixMult(A22, subMatrix(B21, B11, halfSize), halfSize);
    });

    std::future<std::vector<double>> f5 = std::async(std::launch::async, [&]() {
      return strassenMatrixMult(addMatrix(A11, A12, halfSize), B22, halfSize);
    });

    std::future<std::vector<double>> f6 = std::async(std::launch::async, [&]() {
      return strassenMatrixMult(subMatrix(A21, A11, halfSize), addMatrix(B11, B12, halfSize), halfSize);
    });

    std::future<std::vector<double>> f7 = std::async(std::launch::async, [&]() {
      return strassenMatrixMult(subMatrix(A12, A22, halfSize), addMatrix(B21, B22, halfSize), halfSize);
    });

    P1 = f1.get();
    P2 = f2.get();
    P3 = f3.get();
    P4 = f4.get();
    P5 = f5.get();
    P6 = f6.get();
    P7 = f7.get();

    std::vector<double> C11 = addMatrix(subMatrix(addMatrix(P1, P4, halfSize), P5, halfSize), P7, halfSize);
    std::vector<double> C12 = addMatrix(P3, P5, halfSize);
    std::vector<double> C21 = addMatrix(P2, P4, halfSize);
    std::vector<double> C22 = addMatrix(subMatrix(addMatrix(P1, P3, halfSize), P2, halfSize), P6, halfSize);

    C = mergeMatrix(C11, C12, C21, C22);
  }

  return C;
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
