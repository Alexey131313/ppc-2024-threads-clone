// Copyright 2024 Bonyuk Peter
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/bonyuk_p_gauss_gorizont/include/ops_seq.hpp"

TEST(bonyuk_p_gauss_gorizont, test_one) {
  int n = 1000;
  int m = 1000;
  int min = 100;
  int max = 200;

  std::vector<int> image = getImage(n, m, min, max);

  std::vector<int> image_size = {n, m};
  std::vector<int> image_third = {min, max};
  std::vector<int> out(n * m);

  std::vector<int> in(image);
  std::vector<int> result(n * m, 0);

  std::vector<int> gauss_kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  int kSize = 3;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      int sum = 0;
      for (int m = 0; m < kSize; ++m) {
        for (int n = 0; n < kSize; ++n) {
          int row = i + m - kSize / 2;
          int col = j + n - kSize / 2;
          if (row >= 0 && row < n && col >= 0 && col < m) {
            sum += in[row * m + col] * gauss_kernel[m * kSize + n];
          }
        }
      }
      sum = std::min(sum, 255);
      result[i * m + j] = sum;
    }
  }
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image_size.data()));
  task_data->inputs_count.emplace_back(image_size.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data->inputs_count.emplace_back(image.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image_third.data()));
  task_data->inputs_count.emplace_back(image_third.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto test_task = std::make_shared<LinearGaussianFiltering>(task_data);
  auto perf_at = std::make_shared<ppc::core::PerfAttr>();
  perf_at->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_at->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_res = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyz = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyz->pipeline_run(perf_at, perf_res);
  ppc::core::Perf::print_perf_statistic(perf_res);
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(bonyuk_p_gauss_gorizont, test_two) {
  int n = 1000;
  int m = 1000;
  int min = 100;
  int max = 220;

  std::vector<int> image = getImage(n, m, min, max);
  std::vector<int> image_size = {n, m};
  std::vector<int> image_third = {min, max};
  std::vector<int> out(n * m);

  std::vector<int> in(image);
  std::vector<int> result(n * m, 0);
  std::vector<int> gauss_kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  int kSize = 3;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      int sum = 0;
      for (int m = 0; m < kSize; ++m) {
        for (int n = 0; n < kSize; ++n) {
          int row = i + m - kSize / 2;
          int col = j + n - kSize / 2;
          if (row >= 0 && row < n && col >= 0 && col < m) {
            sum += in[row * m + col] * gauss_kernel[m * kSize + n];
          }
        }
      }
      sum = std::min(sum, 255);
      result[i * m + j] = sum;
    }
  }

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image_size.data()));
  task_data->inputs_count.emplace_back(image_size.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data->inputs_count.emplace_back(image.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image_third.data()));
  task_data->inputs_count.emplace_back(image_third.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto test_task = std::make_shared<LinearGaussianFiltering>(task_data);

  auto perf_at = std::make_shared<ppc::core::PerfAttr>();
  perf_at->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_at->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_res = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyz = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyz->task_run(perf_at, perf_res);
  ppc::core::Perf::print_perf_statistic(perf_res);
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(result[i], out[i]);
  }
}
