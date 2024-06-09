// Copyright 2024 Sadikov Damir
#pragma once

#include <complex>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

struct matrix_CRS {
  int n = 0;
  int m = 0;
  std::vector<int> row_id{};
  std::vector<int> col{};
  std::vector<std::complex<double>> value{};
};

class CRSComplexMult_omp : public ppc::core::Task {
 public:
  explicit CRSComplexMult_omp(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  matrix_CRS *A{}, *B{}, *C{};
};
