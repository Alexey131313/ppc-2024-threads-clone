// Copyright 2024 Dostavalov Semyon

#include "seq/dostavalov_s_sop_gradient/include/ops_seq.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace dostavalov_s_seq {
std::vector<double> randVector(int size) {
  std::vector<double> random_vector(size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(MIN_VALUE, MAX_VALUE);

  for (int i = 0; i < size; ++i) {
    random_vector[i] = dis(gen);
  }

  return random_vector;
}

std::vector<double> randMatrix(int size) {
  std::vector<double> random_matrix(size * size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(MIN_VALUE, MAX_VALUE);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      random_matrix[i * size + j] = dis(gen);
    }
  }
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < i; ++j) {
      random_matrix[i * size + j] = random_matrix[j * size + i];
    }
  }

  return random_matrix;
}

bool SeqSLAYGradient::pre_processing() {
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

bool SeqSLAYGradient::validation() {
  internal_order_test();

  if (taskData->inputs_count[0] != taskData->inputs_count[1] * taskData->inputs_count[1]) {
    return false;
  }

  if (taskData->inputs_count[1] != taskData->outputs_count[0]) {
    return false;
  }

  return true;
}

bool SeqSLAYGradient::run() {
  internal_order_test();

  size_t size = vector.size();
  std::vector<double> result(size, 0.0);
  std::vector<double> residual = vector;
  std::vector<double> direction = residual;
  std::vector<double> prev_residual = vector;

  while (true) {
    std::vector<double> A_Dir(size, 0.0);

    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        A_Dir[i] += matrix[i * size + j] * direction[j];
      }
    }

    double residual_dot_residual = 0.0;
    double A_Dir_dot_direction = 0.0;

    for (size_t i = 0; i < size; ++i) {
      residual_dot_residual += residual[i] * residual[i];
      A_Dir_dot_direction += A_Dir[i] * direction[i];
    }

    double alpha = residual_dot_residual / A_Dir_dot_direction;

    for (size_t i = 0; i < result.size(); ++i) {
      result[i] += alpha * direction[i];
    }

    for (size_t i = 0; i < residual.size(); ++i) {
      residual[i] = prev_residual[i] - alpha * A_Dir[i];
    }

    double new_residual = 0.0;

    for (size_t i = 0; i < residual.size(); ++i) {
      new_residual += residual[i] * residual[i];
    }

    if (sqrt(new_residual) < TOLERANCE) {
      break;
    }

    double beta = new_residual / residual_dot_residual;

    for (size_t i = 0; i < size; ++i) {
      direction[i] = residual[i] + beta * direction[i];
    }

    prev_residual = residual;
  }
  answer = result;
  return true;
}

bool SeqSLAYGradient::post_processing() {
  internal_order_test();

  for (size_t i = 0; i < answer.size(); i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = answer[i];
  }
  return true;
}

bool SeqSLAYGradient::check_solution(const std::vector<double>& matrixA, const std::vector<double>& vectorB,
                                     const std::vector<double>& solutionC) {
  bool solution_correct = true;
  size_t size = vectorB.size();
  std::vector<double> A_Sol(size, 0.0);

  for (size_t i = 0; i < size; ++i) {
    A_Sol[i] = 0.0;
    for (size_t j = 0; j < size; ++j) {
      A_Sol[i] += matrixA[i * size + j] * solutionC[j];
    }
  }

  for (size_t i = 0; i < size; ++i) {
    if (std::abs(A_Sol[i] - vectorB[i]) > TOLERANCE) {
      solution_correct = false;
      break;
    }
  }

  return solution_correct;
}

}  // namespace dostavalov_s_seq
