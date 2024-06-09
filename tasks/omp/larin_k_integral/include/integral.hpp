// Copyright 2024 Larin Konstantin
#pragma once

#include <functional>
#include <vector>

// one more useless include
#include "core/task/include/task.hpp"

namespace larin {
using num_t = double;
using state_t = std::vector<num_t>;
using func_t = std::function<num_t(state_t)>;
using limit_t = std::pair<std::function<num_t(state_t)>, std::function<num_t(state_t)>>;

num_t integral(const std::vector<limit_t>& limits, const func_t& func, num_t step);

class useless_class_that_exists_because_of_disgusting_api : public ppc::core::Task {
 public:
  useless_class_that_exists_because_of_disgusting_api(const std::vector<limit_t>& limits, func_t func,
                                                      std::shared_ptr<ppc::core::TaskData> t)
      : Task(std::move(t)), func(std::move(func)), limits(limits) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  num_t get_result() const noexcept { return result_that_cannot_be_returned_from_run; }
  void set_step(num_t stepstepstepstepstepstepstep) { this->step = stepstepstepstepstepstepstep; };

 private:
  num_t result_that_cannot_be_returned_from_run = 0;
  num_t step = 1e-3;
  func_t func;
  std::vector<limit_t> limits;
};
}  // namespace larin
