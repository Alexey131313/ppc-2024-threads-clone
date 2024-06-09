// Copyright 2024 Veselov Ilya
#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace veselov_i_omp {
class SystemsGradMethodOmp : public ppc::core::Task {
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> x;
  int rows;

 public:
  explicit SystemsGradMethodOmp(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
};

bool checkSolution(const std::vector<double> &Aa, const std::vector<double> &bb, const std::vector<double> &xx,
                   double tol = 1e-6);
std::vector<double> genRandomVector(int size, int maxVal);
std::vector<double> genRandomMatrix(int size, int maxVal);
}  // namespace veselov_i_omp
