// Copyright 2024 Safronov Mikhail
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

class SafronovSeqFoxAlgTask : public ppc::core::Task {
 public:
  explicit SafronovSeqFoxAlgTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  double *A{}, *B{}, *C{};
  size_t n{};
};

void GetRandomValue(double *m, int size);
void identityMatrix(double *m, int n, double k = 1.0);
void ModifidentityMatrix(double *m, int n, double k = 1.0);

std::vector<double> mulSafronov(const std::vector<double> &A, const std::vector<double> &B, int n);
