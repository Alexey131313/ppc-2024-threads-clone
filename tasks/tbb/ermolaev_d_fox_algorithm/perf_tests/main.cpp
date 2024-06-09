#include <gtest/gtest.h>
#include <oneapi/tbb.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "tbb/ermolaev_d_fox_algorithm/include/ops_tbb.hpp"

double timer_tbb() {
  static auto start_time_point = oneapi::tbb::tick_count::now();
  return (oneapi::tbb::tick_count::now() - start_time_point).seconds();
}

TEST(ermolaev_d_fox_algorithm_tbb, test_pipline_run) {
  constexpr size_t matrix_size = 512;
  const double tolerance = 1e-5;
  std::mt19937 gen(1);  // Инициализация генератора случайных чисел с начальным значением 1
  std::uniform_real_distribution<> dis(1.0, 6.0);
  std::vector<double> A(matrix_size * matrix_size);
  std::vector<double> B(matrix_size * matrix_size);
  std::vector<double> C(matrix_size * matrix_size, 0.0);
  std::vector<double> expected(matrix_size * matrix_size);

  // Заполнение матриц A и B случайными значениями
  for (size_t i = 0; i < matrix_size * matrix_size; ++i) {
    double random_value = dis(gen);  // Генерация случайного числа
    A[i] = B[i] = random_value;      // Запись случайного числа в матрицы A и B
  }

  // Расчет ожидаемого результата
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      for (size_t k = 0; k < matrix_size; ++k) {
        expected[i * matrix_size + j] += A[i * matrix_size + k] * B[k * matrix_size + j];
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs_count.emplace_back(matrix_size * matrix_size);
  taskData->inputs_count.emplace_back(matrix_size * matrix_size);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskData->outputs_count.emplace_back(matrix_size * matrix_size);

  auto testTask = std::make_shared<FoxAlgorithm>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;
  perfAttr->current_timer = &timer_tbb;

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < matrix_size * matrix_size; i++) {
    ASSERT_NEAR(C[i], expected[i], tolerance);
  }
}
TEST(ermolaev_d_fox_algorithm_tbb, test_task_run) {
  constexpr size_t matrix_size = 512;
  const double tolerance = 1e-5;
  std::mt19937 gen(1);  // Инициализация генератора случайных чисел с начальным значением 1
  std::uniform_real_distribution<> dis(1.0, 6.0);
  std::vector<double> A(matrix_size * matrix_size);
  std::vector<double> B(matrix_size * matrix_size);
  std::vector<double> C(matrix_size * matrix_size, 0.0);
  std::vector<double> expected(matrix_size * matrix_size);
  // Заполнение матриц A и B случайными значениями
  for (size_t i = 0; i < matrix_size * matrix_size; ++i) {
    double random_value = dis(gen);  // Генерация случайного числа
    A[i] = B[i] = random_value;      // Запись случайного числа в матрицы A и B
  }
  // Расчет ожидаемого результата
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      for (size_t k = 0; k < matrix_size; ++k) {
        expected[i * matrix_size + j] += A[i * matrix_size + k] * B[k * matrix_size + j];
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs_count.emplace_back(matrix_size * matrix_size);
  taskData->inputs_count.emplace_back(matrix_size * matrix_size);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(C.data()));
  taskData->outputs_count.emplace_back(matrix_size * matrix_size);

  auto testTask = std::make_shared<FoxAlgorithm>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;
  perfAttr->current_timer = &timer_tbb;

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < matrix_size * matrix_size; i++) {
    ASSERT_NEAR(C[i], expected[i], tolerance);
  }
}