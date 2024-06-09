// Copyright 2023 Korablev Nikita
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "tbb/korablev_n_montecarlo/include/my_funcs.hpp"

class KorablevMonteCarloTBB : public ppc::core::Task {
 public:
  explicit KorablevMonteCarloTBB(std::shared_ptr<ppc::core::TaskData> taskData_, func function)
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