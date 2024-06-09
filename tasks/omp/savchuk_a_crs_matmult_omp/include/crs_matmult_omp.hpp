// Copyright 2024 Savchuk Anton
#pragma once

#include <complex>
#include <vector>

#include "core/task/include/task.hpp"

namespace SavchukOMP {

using Complex = std::complex<double>;

class SavchukCRSMatMultOMPSequential : public ppc::core::Task {
 public:
  explicit SavchukCRSMatMultOMPSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<Complex> values1{};
  std::vector<int> rowPtr1{};
  std::vector<int> colPtr1{};
  int numRows1{};
  int numCols1{};
  std::vector<Complex> values2{};
  std::vector<int> rowPtr2{};
  std::vector<int> colPtr2{};
  int numRows2{};
  int numCols2{};
  Complex *result{};
};

class SavchukCRSMatMultOMPParallel : public ppc::core::Task {
 public:
  explicit SavchukCRSMatMultOMPParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<Complex> values1{};
  std::vector<int> rowPtr1{};
  std::vector<int> colPtr1{};
  int numRows1{};
  int numCols1{};
  std::vector<Complex> values2{};
  std::vector<int> rowPtr2{};
  std::vector<int> colPtr2{};
  int numRows2{};
  int numCols2{};
  Complex *result{};
};
}  // namespace SavchukOMP