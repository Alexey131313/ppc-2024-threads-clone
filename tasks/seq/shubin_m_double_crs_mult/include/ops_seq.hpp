// Copyright 2024 Shubin Mikhail

#pragma once

#include <vector>

#include "core/task/include/task.hpp"
#include "sparsemat_crs.hpp"

class SparseMat_CRS_Mult_Seq : public ppc::core::Task {
  SparseMat_CRS* mat_in_1;
  SparseMat_CRS* mat_in_2;
  SparseMat_CRS* mat_out;

 public:
  explicit SparseMat_CRS_Mult_Seq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
};
