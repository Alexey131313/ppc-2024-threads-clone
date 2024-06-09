// Copyright 2024 kazantsev Evgeny
#pragma once

#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class MatMulStrassenSec : public ppc::core::Task {
 public:
  explicit MatMulStrassenSec(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> result;
  int n = 0, m = 0;
};

inline std::vector<double> getRandomMatrix(int n) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(1.0, 10.0);

  std::vector<double> matrix(n * n);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      matrix[i * n + j] = dis(gen);
    }
  }

  return matrix;
}

std::vector<double> multMatrixNoShtrassen(const std::vector<double>& A, const std::vector<double>& B, int n);
