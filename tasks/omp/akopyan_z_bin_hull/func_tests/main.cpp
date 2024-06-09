// Copyright 2024 Akopyan Zal
#include <gtest/gtest.h>

#include <vector>

#include "omp/akopyan_z_bin_hull/include/ops_omp.hpp"

namespace task = akopyan_z_bin_hull_omp;

TEST(akopyan_z_bin_hull_omp_omp, small_test_tri) {
  int width = 3;
  int height = 4;

  // Create data
  std::vector<int> out(width * height);
  std::vector<int> in = {0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0};

  std::vector<int> hullTrue = {0, 1, 1, 0, 2, 1, task::SEPARATOR};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(width);
  taskData->inputs_count.emplace_back(height);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));

  // Create Task
  task::AkopyanZBinHull testTaskSequential(taskData);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < hullTrue.size(); i++) EXPECT_EQ(hullTrue[i], out[i]);
}

TEST(akopyan_z_bin_hull_omp_omp, small_test_cross) {
  int width = 3;
  int height = 4;

  // Create data
  std::vector<int> out(width * height);
  std::vector<int> in = {0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0};

  std::vector<int> hullTrue = {0, 1, 1, 0, 2, 1, task::SEPARATOR};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(width);
  taskData->inputs_count.emplace_back(height);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));

  // Create Task
  task::AkopyanZBinHull testTaskSequential(taskData);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < hullTrue.size(); i++) EXPECT_EQ(hullTrue[i], out[i]);
}

TEST(akopyan_z_bin_hull_omp_omp, two_labels_test) {
  int width = 6;
  int height = 6;

  // Create data
  std::vector<int> out(width * height);
  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0,
                         0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0};

  std::vector<int> hullTrue = {0, 2, 1, 1, 2, 2, 1, 3, task::SEPARATOR, 3, 4, 4, 2, 5, 4, task::SEPARATOR};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(width);
  taskData->inputs_count.emplace_back(height);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));

  // Create Task
  task::AkopyanZBinHull testTaskSequential(taskData);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < hullTrue.size(); i++) EXPECT_EQ(hullTrue[i], out[i]);
}

TEST(akopyan_z_bin_hull_omp_omp, two_labels_test2) {
  int width = 5;
  int height = 8;

  // Create data
  std::vector<int> out(width * height);
  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0};

  std::vector<int> hullTrue = {2, 2, 3, 1, 4, 2, task::SEPARATOR, 0, 6, 1, 6, 1, 7, 0, 7, task::SEPARATOR};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(width);
  taskData->inputs_count.emplace_back(height);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));

  // Create Task
  task::AkopyanZBinHull testTaskSequential(taskData);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < hullTrue.size(); i++) ASSERT_EQ(hullTrue[i], out[i]);
}

TEST(akopyan_z_bin_hull_omp_omp, generated_test) {
  int width = 10;
  int height = 10;

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
  task::AkopyanZBinHull testTaskSequential(taskData);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < hullTrue.size(); i++) ASSERT_EQ(hullTrue[i], out[i]);
}
