
// Copyright 2024 Martynov Aleksandr
#include "seq/martynov_a_strassen_algorithm/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>
using namespace std::chrono_literals;

inline int get_size(std::vector<double>& a) { return (int)(round(std::sqrt(a.size()))); }

void toSubmatrices(const std::vector<double>& initialMatrix, std::vector<double>& a11, std::vector<double>& a12,
                   std::vector<double>& a21, std::vector<double>& a22) {
  int n = int(std::sqrt(initialMatrix.size()) / 2);

  for (int i = 0; i < n; i++) {
    std::copy(initialMatrix.begin() + (2 * i) * n, initialMatrix.begin() + (2 * i + 1) * n, a11.begin() + i * n);
    std::copy(initialMatrix.begin() + (2 * i + 1) * n, initialMatrix.begin() + (2 * i + 2) * n, a12.begin() + i * n);
    std::copy(initialMatrix.begin() + ((n * n * 2) + (2 * i * n)),
              initialMatrix.begin() + ((n * n * 2) + ((2 * i + 1) * n)), a21.begin() + i * n);
    std::copy(initialMatrix.begin() + ((n * n * 2) + ((2 * i + 1) * n)),
              initialMatrix.begin() + ((n * n * 2) + ((2 * i + 2) * n)), a22.begin() + i * n);
  }
}
inline std::vector<double> tojoin(std::vector<double> a11, std::vector<double> a12, std::vector<double> a21,
                                  std::vector<double> a22) {
  int n = a11.size();
  std::vector<double> result(4 * n, 0.0);
  n = (int)(std::sqrt(n));

  for (int i = 0; i < n; i++) {
    std::copy(a11.begin() + i * n, a11.begin() + (i + 1) * n, result.begin() + (2 * i * n));
    std::copy(a12.begin() + i * n, a12.begin() + (i + 1) * n, result.begin() + (2 * i + 1) * n);
    std::copy(a21.begin() + i * n, a21.begin() + (i + 1) * n, result.begin() + ((n * n * 2) + (2 * i * n)));
    std::copy(a22.begin() + i * n, a22.begin() + (i + 1) * n, result.begin() + ((n * n * 2) + ((2 * i + 1) * n)));
  }
  return result;
}

inline std::vector<double> sum_matrix(const std::vector<double>& a, const std::vector<double>& b) {
  int n = a.size();
  std::vector<double> result(n);

  for (int i = 0; i < n; i++) {
    result[i] = a[i] + b[i];
  }
  return result;
}

inline std::vector<double> sub(const std::vector<double>& a, const std::vector<double>& b) {
  int n = a.size();
  std::vector<double> result(n);

  for (int i = 0; i < n; i++) {
    result[i] = a[i] - b[i];
  }
  return result;
}

bool Strssn_alg::post_processing() {
  internal_order_test();
  std::copy(result.begin(), result.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  return true;
}

bool Strssn_alg::validation() {
  internal_order_test();

  return taskData->inputs_count[1] == taskData->inputs_count[0];
}

bool Strssn_alg::pre_processing() {
  internal_order_test();
  // Init value for input and output
  n = *reinterpret_cast<int*>(taskData->inputs[2]);
  first_matrix = std::vector<double>(taskData->inputs_count[0]);
  second_matrix = std::vector<double>(taskData->inputs_count[1]);

  auto* temporary1 = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* temporary2 = reinterpret_cast<double*>(taskData->inputs[1]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    first_matrix[i] = temporary1[i];
    second_matrix[i] = temporary2[i];
  }
  return true;
}
std::vector<double> strassenM(const std::vector<double>& a, const std::vector<double>& b, int n) {
  int size = n;

  if (size <= 2) {
    return ijkalgorithm(a, b, size);
  }
  size = size / 2;
  std::vector<double> b11(size * size);
  std::vector<double> b12(size * size);
  std::vector<double> b21(size * size);
  std::vector<double> b22(size * size);
  std::vector<double> a11(size * size);
  std::vector<double> a12(size * size);
  std::vector<double> a21(size * size);
  std::vector<double> a22(size * size);

  toSubmatrices(a, a11, a12, a21, a22);
  toSubmatrices(b, b11, b12, b21, b22);

  std::vector<double> p1 = strassenM(sum_matrix(a11, a22), sum_matrix(b11, b22), size);
  std::vector<double> p2 = strassenM(sum_matrix(a21, a22), b11, size);
  std::vector<double> p3 = strassenM(a11, sub(b12, b22), size);
  std::vector<double> p4 = strassenM(a22, sub(b21, b11), size);
  std::vector<double> p5 = strassenM(sum_matrix(a11, a12), b22, size);
  std::vector<double> p6 = strassenM(sub(a21, a11), sum_matrix(b11, b12), size);
  std::vector<double> p7 = strassenM(sub(a12, a22), sum_matrix(b21, b22), size);

  std::vector<double> c11 = sum_matrix(sum_matrix(p1, p4), sub(p7, p5));
  std::vector<double> c12 = sum_matrix(p3, p5);
  std::vector<double> c21 = sum_matrix(p2, p4);
  std::vector<double> c22 = sum_matrix(sub(p1, p2), sum_matrix(p3, p6));
  return tojoin(c11, c12, c21, c22);
}
bool Strssn_alg::run() {
  internal_order_test();
  n = get_size(first_matrix);
  result = strassenM(first_matrix, second_matrix, n);
  return true;
}
std::vector<double> ijkalgorithm(const std::vector<double>& first_matrix, const std::vector<double>& second_matrix,
                                 int n) {
  std::vector<double> result_matrix(n * n, 0.0);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        result_matrix[i * n + j] += first_matrix[i * n + k] * second_matrix[k * n + j];
      }
    }
  }
  return result_matrix;
}
