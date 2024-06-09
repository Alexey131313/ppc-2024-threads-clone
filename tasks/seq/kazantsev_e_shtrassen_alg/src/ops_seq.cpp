// Copyright 2024 Kazantsev Evgeny
#include "seq/kazantsev_e_shtrassen_alg/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<double> multMatrixNoShtrassen(const std::vector<double>& A, const std::vector<double>& B, int n) {
  std::vector<double> C(n * n, 0.0);

  if (n == 0) {
    return std::vector<double>();
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
  return C;
}

// For pre_processing
int log2(int n) {
  int res = 1;
  while ((n >>= 1) != 0) res++;
  return res;
}

int getNewDimention(std::vector<double>& a) {
  int n = std::sqrt(a.size());
  if (n > 0 && (n & (n - 1)) == 0) {
    return n;
  }
  return 1 << log2(n);
}

// add column and rows if not enough
std::vector<double> additionSquareMatrix(std::vector<double>& a, int n) {
  std::vector<double> res(n * n, 0.0);
  int size = std::sqrt(a.size());

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      res[i * size + j] = a[i * size + j];
    }
  }
  return res;
}

// divide into four parts
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

std::vector<double> mergeMatrix1(std::vector<double> a11, std::vector<double> a12, std::vector<double> a21,
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

// summation
std::vector<double> summation(const std::vector<double>& a, const std::vector<double>& b) {
  int n = a.size();
  std::vector<double> res(n);

  for (int i = 0; i < n; i++) {
    res[i] = a[i] + b[i];
  }
  return res;
}

// subtraction
std::vector<double> subtraction(const std::vector<double>& a, const std::vector<double>& b) {
  int n = a.size();
  std::vector<double> res(n);

  for (int i = 0; i < n; i++) {
    res[i] = a[i] - b[i];
  }
  return res;
}

// Algorithm Strassen's
std::vector<double> StrassenMatMul(const std::vector<double>& a, const std::vector<double>& b, int n) {
  int size = n;

  if (size <= 2) {
    return multMatrixNoShtrassen(a, b, size);
  }

  size = size / 2;

  std::vector<double> a11(size * size);
  std::vector<double> a12(size * size);
  std::vector<double> a21(size * size);
  std::vector<double> a22(size * size);

  std::vector<double> b11(size * size);
  std::vector<double> b12(size * size);
  std::vector<double> b21(size * size);
  std::vector<double> b22(size * size);

  splitMatrix(a, a11, a12, a21, a22);
  splitMatrix(b, b11, b12, b21, b22);

  std::vector<double> p1 = StrassenMatMul(summation(a11, a22), summation(b11, b22), size);
  std::vector<double> p2 = StrassenMatMul(summation(a21, a22), b11, size);
  std::vector<double> p3 = StrassenMatMul(a11, subtraction(b12, b22), size);
  std::vector<double> p4 = StrassenMatMul(a22, subtraction(b21, b11), size);
  std::vector<double> p5 = StrassenMatMul(summation(a11, a12), b22, size);
  std::vector<double> p6 = StrassenMatMul(subtraction(a21, a11), summation(b11, b12), size);
  std::vector<double> p7 = StrassenMatMul(subtraction(a12, a22), summation(b21, b22), size);

  std::vector<double> c11 = summation(summation(p1, p4), subtraction(p7, p5));
  std::vector<double> c12 = summation(p3, p5);
  std::vector<double> c21 = summation(p2, p4);
  std::vector<double> c22 = summation(subtraction(p1, p2), summation(p3, p6));
  return mergeMatrix1(c11, c12, c21, c22);
}

bool MatMulStrassenSec::pre_processing() {
  internal_order_test();
  // Init value for input and output
  A = std::vector<double>(taskData->inputs_count[0]);
  B = std::vector<double>(taskData->inputs_count[1]);

  n = *reinterpret_cast<int*>(taskData->inputs[2]);
  m = *reinterpret_cast<int*>(taskData->inputs[3]);

  auto* tmp_ptr_A = reinterpret_cast<double*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    A[i] = tmp_ptr_A[i];
  }

  auto* tmp_ptr_B = reinterpret_cast<double*>(taskData->inputs[1]);
  for (unsigned i = 0; i < taskData->inputs_count[1]; i++) {
    B[i] = tmp_ptr_B[i];
  }
  return true;
}

bool MatMulStrassenSec::validation() {
  internal_order_test();
  // Check count elements of output
  if (taskData->inputs_count[0] != taskData->inputs_count[1]) return false;
  if (taskData->inputs_count[0] % 2 != 0) return false;
  if (taskData->inputs_count[1] % 2 != 0) return false;
  return true;
}

bool MatMulStrassenSec::run() {
  internal_order_test();
  n = getNewDimention(A);
  result = StrassenMatMul(A, B, n);
  return true;
}

bool MatMulStrassenSec::post_processing() {
  internal_order_test();
  std::copy(result.begin(), result.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  return true;
}
