// Copyright 2024 Ustinov Alexander
#pragma once

#include "core/task/include/task.hpp"
#include "tbb/ustinov_a_spgemm_csc_complex/include/sparse_matrix.hpp"

class SpgemmCSCComplexTBBSeq : public ppc::core::Task {
 public:
  explicit SpgemmCSCComplexTBBSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  sparse_matrix *A, *B, *C;
};

class SpgemmCSCComplexTBBPar : public ppc::core::Task {
 public:
  explicit SpgemmCSCComplexTBBPar(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  sparse_matrix *A, *B, *C;
};