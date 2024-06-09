// Copyright 2024 Dostavalov Semyon
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/dostavalov_s_sop_gradient/include/ops_seq.hpp"

using namespace dostavalov_s_seq;
const int SIZE = 300;

std::shared_ptr<ppc::core::TaskData> createTaskData(std::vector<double> &matrix, std::vector<double> &vector,
                                                    std::vector<double> &result) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  return taskDataSeq;
}

std::shared_ptr<ppc::core::PerfAttr> start_performance_timer() {
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();

  perfAttr->current_timer = [=] {
    auto current_time_point = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();

    return static_cast<double>(duration) * 1e-9;
  };

  return perfAttr;
}

TEST(dostavalov_s_sop_gradient, test_pipeline) {
  std::vector<double> matrix = randMatrix(SIZE);
  std::vector<double> vector = randVector(SIZE);
  std::vector<double> result(SIZE);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = createTaskData(matrix, vector, result);

  auto testTaskSeq = std::make_shared<SeqSLAYGradient>(taskDataSeq);
  SeqSLAYGradient testTaskSequential(taskDataSeq);

  auto perfAttr = start_performance_timer();

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSeq);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_TRUE(testTaskSequential.check_solution(matrix, vector, result));
}

TEST(dostavalov_s_sop_gradient, test_task_run) {
  std::vector<double> matrix = randMatrix(SIZE);
  std::vector<double> vector = randVector(SIZE);
  std::vector<double> result(SIZE);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = createTaskData(matrix, vector, result);

  auto testTaskSeq = std::make_shared<SeqSLAYGradient>(taskDataSeq);
  SeqSLAYGradient testTaskSequential(taskDataSeq);

  auto perfAttr = start_performance_timer();

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSeq);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_TRUE(testTaskSequential.check_solution(matrix, vector, result));
}  // namespace dostavalov_s_seq