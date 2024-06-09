// Copyright 2024 Kostin Artem
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class ConjugateGradientMethodSequential : public ppc::core::Task {
 public:
  explicit ConjugateGradientMethodSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A;
  int size = 0;
  std::vector<double> b;
  std::vector<double> x;
};

std::vector<double> generateSPDMatrix(int size, int max_value);

std::vector<double> generatePDVector(int size, int max_value);

bool check_solution(const std::vector<double>& A, int n, const std::vector<double>& b, const std::vector<double>& x,
                    double tolerance);
