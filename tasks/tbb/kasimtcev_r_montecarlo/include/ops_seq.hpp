// Copyright 2024 Kasimtcev Roman
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/kasimtcev_r_montecarlo/include/my_funcs.hpp"

class KasimtcevSequentialMonteCarlo : public ppc::core::Task {
 public:
  explicit KasimtcevSequentialMonteCarlo(std::shared_ptr<ppc::core::TaskData> taskData_, func function)
      : Task(std::move(taskData_)), function(function) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  func function{};
  double Int1[2]{}, Int2[2]{};

  int N = 100;
  double res{};
};