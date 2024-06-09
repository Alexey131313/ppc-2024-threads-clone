// Copyright 2024 Dostavalov Semyon

#include "tbb/dostavalov_s_sop_gradient/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <atomic>
#include <cmath>
#include <random>
#include <vector>

namespace dostavalov_s_tbb {
std::vector<double> randVector(int size) {
  std::vector<double> random_vector(size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(MIN_VALUE, MAX_VALUE);

  tbb::parallel_for(tbb::blocked_range<int>(0, size), [&](const tbb::blocked_range<int>& range) {
    for (int i = range.begin(); i != range.end(); ++i) {
      random_vector[i] = dis(gen);
    }
  });

  return random_vector;
}

std::vector<double> randMatrix(int size) {
  std::vector<double> random_matrix(size * size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(MIN_VALUE, MAX_VALUE);

  tbb::parallel_for(tbb::blocked_range<int>(0, size), [&](const tbb::blocked_range<int>& range) {
    for (int i = range.begin(); i != range.end(); ++i) {
      for (int j = i; j < size; ++j) {
        double value = dis(gen);
        random_matrix[i * size + j] = value;
        random_matrix[j * size + i] = value;
      }
    }
  });

  return random_matrix;
}

std::atomic<double>& operator+=(std::atomic<double>& atomic_value, double value) {
  double current_value = atomic_value.load();
  while (!atomic_value.compare_exchange_weak(current_value, current_value + value))
    ;
  return atomic_value;
}

bool TbbSLAYGradient::pre_processing() {
  internal_order_test();

  const double* input_data_A = reinterpret_cast<double*>(taskData->inputs[0]);

  matrix.resize(taskData->inputs_count[0]);
  std::copy(input_data_A, input_data_A + taskData->inputs_count[0], matrix.begin());

  const double* input_data_B = reinterpret_cast<double*>(taskData->inputs[1]);

  vector.resize(taskData->inputs_count[1]);
  std::copy(input_data_B, input_data_B + taskData->inputs_count[1], vector.begin());

  answer.resize(vector.size(), 0);

  return true;
}

bool TbbSLAYGradient::validation() {
  internal_order_test();

  if (taskData->inputs_count[0] != taskData->inputs_count[1] * taskData->inputs_count[1]) {
    return false;
  }

  if (taskData->inputs_count[1] != taskData->outputs_count[0]) {
    return false;
  }

  return true;
}

bool TbbSLAYGradient::run() {
  internal_order_test();

  long size = vector.size();
  std::vector<double> result(size, 0.0);
  std::vector<double> residual = vector;
  std::vector<double> direction = residual;
  std::vector<double> prev_residual = vector;

  double* matrix_ptr = matrix.data();

  while (true) {
    std::vector<double> A_Dir(size, 0.0);

    tbb::parallel_for(tbb::blocked_range<long>(0, size), [&](const tbb::blocked_range<long>& range) {
      for (long i = range.begin(); i != range.end(); ++i) {
        for (long j = 0; j < size; ++j) {
          A_Dir[i] += matrix_ptr[i * size + j] * direction[j];
        }
      }
    });

    double residual_dot_residual = computeDotProduct(residual, residual);
    double A_Dir_dot_direction = computeDotProduct(A_Dir, direction);
    double alpha = residual_dot_residual / A_Dir_dot_direction;

    updateResult(result, direction, alpha);

    updateResidual(residual, prev_residual, A_Dir, alpha);

    double new_residual = computeDotProduct(residual, residual);

    if (sqrt(new_residual) < TOLERANCE) {
      break;
    }

    double beta = new_residual / residual_dot_residual;

    updateDirection(direction, residual, beta);

    prev_residual = residual;
  }

  answer = result;
  return true;
}

double TbbSLAYGradient::computeDotProduct(const std::vector<double>& vec1, const std::vector<double>& vec2) {
  double dot_product = 0.0;
  for (size_t i = 0; i < vec1.size(); ++i) {
    dot_product += vec1[i] * vec2[i];
  }
  return dot_product;
}

void TbbSLAYGradient::updateResult(std::vector<double>& result, const std::vector<double>& direction, double alpha) {
  std::vector<std::atomic<double>> atomic_result(result.begin(), result.end());

  tbb::parallel_for(tbb::blocked_range<long>(0, static_cast<long>(result.size())),
                    [&](const tbb::blocked_range<long>& range) {
                      for (long i = range.begin(); i != range.end(); ++i) {
                        atomic_result[i] += alpha * direction[i];
                      }
                    });

  for (long i = 0; i < static_cast<long>(result.size()); ++i) {
    result[i] = atomic_result[i];
  }
}

void TbbSLAYGradient::updateResidual(std::vector<double>& residual, const std::vector<double>& prev_residual,
                                     const std::vector<double>& A_Dir, double alpha) {
  std::vector<std::atomic<double>> atomic_residual(residual.begin(), residual.end());

  tbb::parallel_for(tbb::blocked_range<long>(0, static_cast<long>(residual.size())),
                    [&](const tbb::blocked_range<long>& range) {
                      for (long i = range.begin(); i != range.end(); ++i) {
                        atomic_residual[i] = prev_residual[i] - alpha * A_Dir[i];
                      }
                    });

  for (long i = 0; i < static_cast<long>(residual.size()); ++i) {
    residual[i] = atomic_residual[i];
  }
}

void TbbSLAYGradient::updateDirection(std::vector<double>& direction, const std::vector<double>& residual,
                                      double beta) {
  std::vector<std::atomic<double>> atomic_direction(direction.begin(), direction.end());

  tbb::parallel_for(tbb::blocked_range<long>(0, static_cast<long>(direction.size())),
                    [&](const tbb::blocked_range<long>& range) {
                      for (long i = range.begin(); i != range.end(); ++i) {
                        atomic_direction[i] = residual[i] + beta * direction[i];
                      }
                    });

  for (long i = 0; i < static_cast<long>(direction.size()); ++i) {
    direction[i] = atomic_direction[i];
  }
}

bool TbbSLAYGradient::post_processing() {
  internal_order_test();
  long size = answer.size();

  tbb::parallel_for(tbb::blocked_range<long>(0, size), [&](const tbb::blocked_range<long>& range) {
    for (long i = range.begin(); i != range.end(); ++i) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = answer[i];
    }
  });

  return true;
}

bool check_solution(const std::vector<double>& matrixA, const std::vector<double>& vectorB,
                    const std::vector<double>& solutionC) {
  bool solution_correct = true;

  long size = vectorB.size();
  std::vector<double> A_Sol(size, 0.0);

  tbb::parallel_for(tbb::blocked_range<long>(0, size), [&](const tbb::blocked_range<long>& range) {
    for (long i = range.begin(); i != range.end(); ++i) {
      A_Sol[i] = 0.0;
      for (long j = 0; j < size; ++j) {
        A_Sol[i] += matrixA[i * size + j] * solutionC[j];
      }
    }
  });

  tbb::parallel_for(tbb::blocked_range<long>(0, size), [&](const tbb::blocked_range<long>& range) {
    for (long i = range.begin(); i != range.end(); ++i) {
      double A_Sol_i = 0.0;
      for (long j = 0; j < size; ++j) {
        A_Sol_i += matrixA[i * size + j] * solutionC[j];
      }
      if (!solution_correct && std::abs(A_Sol_i - vectorB[i]) > TOLERANCE) {
        solution_correct = false;
        return;
      }
    }
  });

  return solution_correct;
}

}  // namespace dostavalov_s_tbb
