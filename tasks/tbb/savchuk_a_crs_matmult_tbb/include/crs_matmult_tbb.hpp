// Copyright 2024 Savchuk Anton
#pragma once

#include <complex>
#include <vector>

#include "core/task/include/task.hpp"

namespace SavchukTbb {

using Complex = std::complex<double>;

class SavchukCRSMatMultTBBSequential : public ppc::core::Task {
 public:
  explicit SavchukCRSMatMultTBBSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
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

class SavchukCRSMatMultTBBParallel : public ppc::core::Task {
 public:
  explicit SavchukCRSMatMultTBBParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
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
}  // namespace SavchukTbb