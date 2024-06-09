// Copyright 2023 Sredneva Anastasiya
#ifndef TASKS_EXAMPLES_TEST_TBB_OPS_TBB_H_
#define TASKS_EXAMPLES_TEST_TBB_OPS_TBB_H_

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

std::vector<uint8_t> getRandomPicture(int n, int m, uint8_t min, uint8_t max);

class ContrastEnhancement_TBB_Sequential : public ppc::core::Task {
 public:
  explicit ContrastEnhancement_TBB_Sequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<uint8_t> input_ = {};
  std::vector<uint8_t> res = {};
  int n{}, m{};
  uint8_t min{}, max{};
};

class ContrastEnhancement_TBB_Parallel : public ppc::core::Task {
 public:
  explicit ContrastEnhancement_TBB_Parallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<uint8_t> input_ = {};
  std::vector<uint8_t> res = {};
  int n{}, m{};
  uint8_t min{}, max{};
};

#endif  // TASKS_EXAMPLES_TEST_TBB_OPS_TBB_H_
