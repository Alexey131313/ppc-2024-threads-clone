// Copyright 2024 Kirillov Maxim
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

class StrassenMatrixMultSequential : public ppc::core::Task {
 public:
  explicit StrassenMatrixMultSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A, B, C;
  int n = 0;
};

std::vector<double> strassenKirillov(const std::vector<double>& A, const std::vector<double>& B, int n);
std::vector<double> addKirillov(const std::vector<double>& A, const std::vector<double>& B);
std::vector<double> subKirillov(const std::vector<double>& A, const std::vector<double>& B);
std::vector<double> mulKirillov(const std::vector<double>& A, const std::vector<double>& B, int n);
void splitMatrixKirillov(const std::vector<double>& A, std::vector<double>& A11, std::vector<double>& A12,
                         std::vector<double>& A21, std::vector<double>& A22);
std::vector<double> joinMatricesKirillov(const std::vector<double>& A11, const std::vector<double>& A12,
                                         const std::vector<double>& A21, const std::vector<double>& A22, int n);
std::vector<double> generateRandomMatrixKirillov(int n);
