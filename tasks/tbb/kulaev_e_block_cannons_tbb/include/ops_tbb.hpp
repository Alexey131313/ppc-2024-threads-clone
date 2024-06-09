// Copyright 2024 Kulaev Zhenya
#pragma once

#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class TestTBBSequentialKulaevCannon : public ppc::core::Task {
 public:
  explicit TestTBBSequentialKulaevCannon(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
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

std::vector<double> getRandomMatrix(int rows, int cols);

std::vector<double> multiplyMatrix(const std::vector<double>& A, const std::vector<double>& B, int rows_A, int col_B);

std::vector<double> cannonMatrixMultiplication(const std::vector<double>& A, const std::vector<double>& B, int n,
                                               int m);

std::vector<double> cannonMatrixMultiplication_tbb(const std::vector<double>& A, const std::vector<double>& B, int n,
                                                   int m);

class TestTaskTBBParallelKulaevCannon : public ppc::core::Task {
 public:
  explicit TestTaskTBBParallelKulaevCannon(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
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
