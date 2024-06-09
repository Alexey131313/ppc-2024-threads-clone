
// Copyright 2024 Martynov Aleksandr
#pragma once
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
inline std::vector<double> fillMatrix(int n) {
  std::random_device seed;
  std::mt19937 geerator(seed());
  std::uniform_real_distribution<double> random_number(-10.0, 10.0);
  std::vector<double> result(n * n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      result[i * n + j] = random_number(geerator);
    }
  }
  return result;
}
std::vector<double> ijkalgorithm(const std::vector<double>& first_matrix, const std::vector<double>& second_matrix,
                                 int n);

class Strssn_alg : public ppc::core::Task {
 public:
  explicit Strssn_alg(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int n = 0;
  std::vector<double> first_matrix;
  std::vector<double> second_matrix;
  std::vector<double> result;
};
