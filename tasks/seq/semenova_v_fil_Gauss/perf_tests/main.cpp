// Copyright 2024 Semenova Veronika
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/semenova_v_fil_Gauss/include/ops_seq.hpp"

TEST(semenova_v_fil_Gauss_seq, test_pipeline_run) {
  int n = 6500;
  int m = 6500;
  int* image = new int[n * m];
  int* filteredImage = new int[n * m];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (i == j) {
        image[i * m + j] = 10;
        filteredImage[i * m + j] = 10;
      } else {
        image[i * m + j] = 0;
        filteredImage[i * m + j] = 0;
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(filteredImage));
  taskDataSeq->outputs_count.emplace_back(n);
  taskDataSeq->outputs_count.emplace_back(m);

  auto testTaskSeq = std::make_shared<ImageFilGauss>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSeq);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (int i = 1; i < n - 1; ++i) {
    for (int j = 1; j < m - 1; ++j) {
      if (i == j) {
        EXPECT_EQ(filteredImage[i * m + j], 3);
      } else {
        if (i == j + 1 || j == i + 1) {
          EXPECT_EQ(filteredImage[i * m + j], 2);
        } else {
          EXPECT_EQ(filteredImage[i * m + j], 0);
        }
      }
    }
  }
  delete[] image;
  delete[] filteredImage;
}

TEST(semenova_v_fil_Gauss_seq, test_task_run) {
  int n = 6500;
  int m = 6500;
  int* image = new int[n * m];
  int* filteredImage = new int[n * m];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (i == j) {
        image[i * m + j] = 10;
        filteredImage[i * m + j] = 10;
      } else {
        image[i * m + j] = 0;
        filteredImage[i * m + j] = 0;
      }
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(image));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(filteredImage));
  taskDataSeq->outputs_count.emplace_back(n);
  taskDataSeq->outputs_count.emplace_back(m);

  auto testTaskSeq = std::make_shared<ImageFilGauss>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSeq);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (int i = 1; i < n - 1; ++i) {
    for (int j = 1; j < m - 1; ++j) {
      if (i == j) {
        EXPECT_EQ(filteredImage[i * m + j], 3);
      } else {
        if (i == j + 1 || j == i + 1) {
          EXPECT_EQ(filteredImage[i * m + j], 2);
        } else {
          EXPECT_EQ(filteredImage[i * m + j], 0);
        }
      }
    }
  }
  delete[] image;
  delete[] filteredImage;
}
