// Copyright 2024 Larin Konstantin
#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/larin_k_integral/include/integral.hpp"

using namespace larin;

TEST(larin_k_integral, test_pipeline_run) {
  // Create data

  limit_t phi_limits{func_t{[](auto &&) { return 0; }}, func_t{[](auto &&) { return M_PI; }}};
  limit_t theta_limits{func_t{[](auto &&) { return 0; }}, func_t{[](auto &&) { return M_PI; }}};
  limit_t r_limits{func_t{[](auto &&) { return 0; }}, func_t{[](auto &&) { return 1; }}};

  func_t func = [](const std::vector<num_t> &x) {
    auto r = x[2];
    auto theta = x[1];
    return r * r * std::sin(theta);
  };

  num_t expected_value = 2 / 3. * M_PI;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  // Create Task
  auto testTaskSequential = std::make_shared<useless_class_that_exists_because_of_disgusting_api>(
      std::vector<limit_t>{phi_limits, theta_limits, r_limits}, func, taskDataSeq);

  testTaskSequential->set_step(1e-2);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = []() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    return static_cast<double>(current_time_point.time_since_epoch().count()) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  EXPECT_NEAR(testTaskSequential->get_result(), expected_value, 1e-2);
}

TEST(larin_k_integral, test_task_run) {
  // Create data

  limit_t phi_limits{func_t{[](auto &&) { return 0; }}, func_t{[](auto &&) { return M_PI; }}};
  limit_t theta_limits{func_t{[](auto &&) { return 0; }}, func_t{[](auto &&) { return M_PI; }}};
  limit_t r_limits{func_t{[](auto &&) { return 0; }}, func_t{[](auto &&) { return 1; }}};

  func_t func = [](const std::vector<num_t> &x) {
    auto r = x[2];
    auto theta = x[1];
    return r * r * std::sin(theta);
  };

  num_t expected_value = 2 / 3. * M_PI;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  // Create Task
  auto testTaskSequential = std::make_shared<useless_class_that_exists_because_of_disgusting_api>(
      std::vector<limit_t>{phi_limits, theta_limits, r_limits}, func, taskDataSeq);

  testTaskSequential->set_step(1e-2);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = []() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    return static_cast<double>(current_time_point.time_since_epoch().count()) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  EXPECT_NEAR(testTaskSequential->get_result(), expected_value, 1e-2);
}
