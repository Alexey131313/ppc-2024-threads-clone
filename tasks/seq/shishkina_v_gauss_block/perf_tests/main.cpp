// Copyright 2023 Shishkina Valeriya
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/shishkina_v_gauss_block/include/ops_seq.hpp"

TEST(shishkina_v_gauss_block, test_pipeline_run) {
  int height = 1000;
  int width = 1000;
  int min = 40;
  int max = 180;

  // Create data
  std::vector<int> in = getPicture(height, width, min, max);

  std::vector<int> in2 = {height, width};
  std::vector<int> in3 = {min, max};
  std::vector<int> out(height * width);

  std::vector<int> input(in);
  std::vector<int> res(height * width, 0);

  int blockSize = std::sqrt(input.size());
  std::vector<int> gaussianKernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  int kernelSize = 3;
  for (int i = 0; i < height; i += blockSize) {
    for (int j = 0; j < width; j += blockSize) {
      for (int x = i; x < std::min(i + blockSize, height); ++x) {
        for (int y = j; y < std::min(j + blockSize, width); ++y) {
          int sum = 0;
          for (int m = 0; m < kernelSize; ++m) {
            for (int n = 0; n < kernelSize; ++n) {
              int row = x + m - kernelSize / 2;
              int col = y + n - kernelSize / 2;
              if (row >= 0 && row < height && col >= 0 && col < width) {
                sum += input[row * width + col] * gaussianKernel[m * kernelSize + n];
              }
            }
          }
          sum = std::min(sum, 255);
          res[x * width + y] = sum;
        }
      }
    }
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(in2.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataSeq->inputs_count.emplace_back(in3.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<LinearFilteringGauss>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (int i = 0; i < height * width; i++) {
    ASSERT_EQ(res[i], out[i]);
  }
}
TEST(shishkina_v_gauss_block, test_task_run) {
  int height = 1000;
  int width = 1000;
  int min = 40;
  int max = 180;

  // Create data
  std::vector<int> in_ = getPicture(height, width, min, max);

  std::vector<int> tmp_sec = {height, width};
  std::vector<int> tmp_third = {min, max};
  std::vector<int> out(height * width);

  std::vector<int> input(in_);
  std::vector<int> res(height * width, 0);
  int blockSize = std::sqrt(input.size());
  std::vector<int> gaussianKernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  int kernelSize = 3;
  for (int i = 0; i < height; i += blockSize) {
    for (int j = 0; j < width; j += blockSize) {
      for (int x = i; x < std::min(i + blockSize, height); ++x) {
        for (int y = j; y < std::min(j + blockSize, width); ++y) {
          int sum = 0;
          for (int m = 0; m < kernelSize; ++m) {
            for (int n = 0; n < kernelSize; ++n) {
              int row = x + m - kernelSize / 2;
              int col = y + n - kernelSize / 2;
              if (row >= 0 && row < height && col >= 0 && col < width) {
                sum += input[row * width + col] * gaussianKernel[m * kernelSize + n];
              }
            }
          }
          sum = std::min(sum, 255);
          res[x * width + y] = sum;
        }
      }
    }
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_sec.data()));
  taskDataSeq->inputs_count.emplace_back(tmp_sec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_.data()));
  taskDataSeq->inputs_count.emplace_back(in_.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_third.data()));
  taskDataSeq->inputs_count.emplace_back(tmp_third.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<LinearFilteringGauss>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (int i = 0; i < height * width; i++) {
    ASSERT_EQ(res[i], out[i]);
  }
}