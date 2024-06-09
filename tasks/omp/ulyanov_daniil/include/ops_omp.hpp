// Copyright 2024 Ulyanov Daniil
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class OMPFilterGaussHorizontalSeqUlyanov : public ppc::core::Task {
 public:
  explicit OMPFilterGaussHorizontalSeqUlyanov(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<uint8_t> inputImage;
  std::vector<uint8_t> resultImage;
  std::vector<float> kernel;
  int width, height;
};

class OMPFilterGaussHorizontalParUlyanov : public ppc::core::Task {
 public:
  explicit OMPFilterGaussHorizontalParUlyanov(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<uint8_t> inputImage;
  std::vector<uint8_t> resultImage;
  std::vector<float> kernel;
  int width, height;
};
