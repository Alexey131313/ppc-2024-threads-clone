// Copyright 2024 Akopyan Zal
#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/akopyan_z_bin_hull/include/ops_omp.hpp"

namespace task = akopyan_z_bin_hull_omp;

TEST(akopyan_z_bin_hull_omp_omp, test_pipeline_run) {
  int width = 2000;
  int height = 2000;

  // Create data
  std::vector<int> out(width * height);
  std::vector<int> in(width * height, 0);
  int sqw = width / 2;
  int sqh = height / 2;
  for (int i = 0; i < sqh - 1; i++) {
    for (int j = 0; j < sqw - 1; j++) {
      in[i * width + j] = 1;
      in[i * width + (sqw + 1 + j)] = 1;
    }
  }
  for (int i = 0; i < width; i++) {
    in[sqh * width + i] = 1;
    in[(sqh + 2) * width + i] = 1;
  }
  for (int i = sqh + 3; i < height; i++) {
    in[i * width + sqw - 1] = 1;
    in[i * width + sqw] = 1;
  }

  std::vector<int> hullTrue = {
      0,       0,          sqw - 2,   0,          sqw - 2,         sqh - 2, 0,       sqh - 2,   task::SEPARATOR,
      sqw + 1, 0,          width - 1, 0,          width - 1,       sqh - 2, sqw + 1, sqh - 2,   task::SEPARATOR,
      0,       sqh,        width - 1, sqh,        task::SEPARATOR, 0,       sqh + 2, width - 1, sqh + 2,
      sqw,     height - 1, sqw - 1,   height - 1, task::SEPARATOR};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(width);
  taskData->inputs_count.emplace_back(height);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  // Create Task
  auto testTaskSequential = std::make_shared<task::AkopyanZBinHull>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < hullTrue.size(); i++) ASSERT_EQ(hullTrue[i], out[i]);
}

TEST(akopyan_z_bin_hull_omp_omp, test_task_run) {
  int width = 2000;
  int height = 2000;

  // Create data
  std::vector<int> out(width * height);
  std::vector<int> in(width * height, 0);
  int sqw = width / 2;
  int sqh = height / 2;
  for (int i = 0; i < sqh - 1; i++) {
    for (int j = 0; j < sqw - 1; j++) {
      in[i * width + j] = 1;
      in[i * width + (sqw + 1 + j)] = 1;
    }
  }
  for (int i = 0; i < width; i++) {
    in[sqh * width + i] = 1;
    in[(sqh + 2) * width + i] = 1;
  }
  for (int i = sqh + 3; i < height; i++) {
    in[i * width + sqw - 1] = 1;
    in[i * width + sqw] = 1;
  }

  std::vector<int> hullTrue = {
      0,       0,          sqw - 2,   0,          sqw - 2,         sqh - 2, 0,       sqh - 2,   task::SEPARATOR,
      sqw + 1, 0,          width - 1, 0,          width - 1,       sqh - 2, sqw + 1, sqh - 2,   task::SEPARATOR,
      0,       sqh,        width - 1, sqh,        task::SEPARATOR, 0,       sqh + 2, width - 1, sqh + 2,
      sqw,     height - 1, sqw - 1,   height - 1, task::SEPARATOR};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(width);
  taskData->inputs_count.emplace_back(height);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  // Create Task
  auto testTaskSequential = std::make_shared<task::AkopyanZBinHull>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < hullTrue.size(); i++) ASSERT_EQ(hullTrue[i], out[i]);
}
