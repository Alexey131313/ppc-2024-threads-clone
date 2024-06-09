// Copyright 2024 Kulagin Aleksandr
#pragma once

#include "core/task/include/task.hpp"
#include "seq/kulagin_a_gauss_filter_vert/include/common.hpp"

class FilterGaussVerticalTaskSequential : public ppc::core::Task {
 public:
  explicit FilterGaussVerticalTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  uint32_t* img{};
  size_t w{}, h{};
  float* kernel{};
  std::unique_ptr<uint32_t[]> img_res;
};
