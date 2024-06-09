// Copyright 2024 Vinichuk Timofey
#pragma once

#include "core/task/include/task.hpp"
#include "seq/vinicuk_t_complex_sparse_matrix_mult_csc/include/csc_complex_matrix.hpp"

class MultMatrixCSCComplex : public ppc::core::Task {
 public:
  explicit MultMatrixCSCComplex(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  CSCComplexMatrix *mtrx_A, *mtrx_B, *mtrx_res;
};