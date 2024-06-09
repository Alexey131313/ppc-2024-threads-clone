// Copyright 2024 Kasimtcev Roman
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/kasimtcev_r_montecarlo/include/my_funcs.hpp"

class KasimtcevSequentialMonteCarlo : public ppc::core::Task {
 public:
  explicit KasimtcevSequentialMonteCarlo(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  kasimtcev_func function{};
  double Int1[2]{}, Int2[2]{};

  int N{};
  double res{};
};
