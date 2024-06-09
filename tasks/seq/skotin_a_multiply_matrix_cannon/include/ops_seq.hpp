// Copyright 2024 Skotin Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class MatrixMultiplicationTask : public ppc::core::Task {
 public:
  explicit MatrixMultiplicationTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> matrixA;
  std::vector<std::vector<double>> matrixB;
  std::vector<std::vector<double>> resultMatrix;

  static bool loadMatrix(const std::vector<double>& inputData, std::vector<std::vector<double>>& matrix, size_t size);
  bool saveResult();
};
