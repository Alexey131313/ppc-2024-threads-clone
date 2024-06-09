// Copyright 2024 Dostavalov Semyons

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace dostavalov_s_tbb {
const double TOLERANCE = 0.0001;
const double MIN_VALUE = 0.0;
const double MAX_VALUE = 50.0;

std::vector<double> randVector(int size);
std::vector<double> randMatrix(int size);
bool check_solution(const std::vector<double>& matrixA, const std::vector<double>& vectorB,
                    const std::vector<double>& solutionC);

class TbbSLAYGradient : public ppc::core::Task {
 public:
  explicit TbbSLAYGradient(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  double static computeDotProduct(const std::vector<double>& vec1, const std::vector<double>& vec2);
  void static updateResult(std::vector<double>& result, const std::vector<double>& direction, double alpha);
  void static updateResidual(std::vector<double>& residual, const std::vector<double>& prev_residual,
                             const std::vector<double>& A_Dir, double alpha);
  void static updateDirection(std::vector<double>& direction, const std::vector<double>& residual, double beta);

 private:
  std::vector<double> matrix, vector, answer;
};

}  // namespace dostavalov_s_tbb
