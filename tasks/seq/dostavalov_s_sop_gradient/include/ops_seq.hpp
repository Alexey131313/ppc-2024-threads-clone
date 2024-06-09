// Copyright 2024 Dostavalov Semyon

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace dostavalov_s_seq {
const double TOLERANCE = 0.0001;
const double MIN_VALUE = 0.0;
const double MAX_VALUE = 50.0;

std::vector<double> randVector(int size);
std::vector<double> randMatrix(int size);

class SeqSLAYGradient : public ppc::core::Task {
 public:
  explicit SeqSLAYGradient(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static bool check_solution(const std::vector<double>& matrixA, const std::vector<double>& vectorB,
                             const std::vector<double>& solutionC);

 private:
  std::vector<double> matrix, vector, answer;
};

}  // namespace dostavalov_s_seq
