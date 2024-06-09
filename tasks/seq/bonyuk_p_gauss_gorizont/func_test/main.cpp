// Copyright 2024 Bonyuk Peter
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "seq/bonyuk_p_gauss_gorizont/include/ops_seq.hpp"

TEST(bonyuk_p_gauss_gorizont, one_test) {
  int n = 3;
  int m = 3;

  std::vector<int> image = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  int min = *std::min_element(image.begin(), image.end());
  int max = *std::max_element(image.begin(), image.end());

  std::vector<int> image_size = {n, m};
  std::vector<int> image_third = {min, max};
  std::vector<int> out(n * m);

  std::vector<int> result = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image_size.data()));
  task_data->inputs_count.emplace_back(image_size.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data->inputs_count.emplace_back(image.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image_third.data()));
  task_data->inputs_count.emplace_back(image_third.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  LinearGaussianFiltering test_task(task_data);
  ASSERT_EQ(test_task.validation(), true);
  test_task.pre_processing();
  ASSERT_EQ(test_task.run(), true);
  test_task.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(bonyuk_p_gauss_gorizont, two_test) {
  int n = 3;
  int m = 3;

  std::vector<int> image = {255, 255, 255, 255, 255, 255, 255, 255, 255};

  int min = *std::min_element(image.begin(), image.end());
  int max = *std::max_element(image.begin(), image.end());

  std::vector<int> image_size = {n, m};
  std::vector<int> image_third = {min, max};
  std::vector<int> out(n * m);

  std::vector<int> result = {255, 255, 255, 255, 255, 255, 255, 255, 255};

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image_size.data()));
  task_data->inputs_count.emplace_back(image_size.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data->inputs_count.emplace_back(image.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image_third.data()));
  task_data->inputs_count.emplace_back(image_third.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  LinearGaussianFiltering test_task(task_data);
  ASSERT_EQ(test_task.validation(), true);
  test_task.pre_processing();
  ASSERT_EQ(test_task.run(), true);
  test_task.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(result[i], out[i]);
  }
}
TEST(bonyuk_p_gauss_gorizont, three_test) {
  int n = 3;
  int m = 3;

  std::vector<int> image = {100, 150, 200, 50, 100, 150, 0, 50, 100};

  int min = *std::min_element(image.begin(), image.end());
  int max = *std::max_element(image.begin(), image.end());

  std::vector<int> image_size = {n, m};
  std::vector<int> image_third = {min, max};
  std::vector<int> out(n * m);

  std::vector<int> result = {100, 150, 200, 75, 125, 175, 31, 57, 82};

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image_size.data()));
  task_data->inputs_count.emplace_back(image_size.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data->inputs_count.emplace_back(image.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image_third.data()));
  task_data->inputs_count.emplace_back(image_third.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  LinearGaussianFiltering test_task(task_data);
  ASSERT_EQ(test_task.validation(), true);
  test_task.pre_processing();
  ASSERT_EQ(test_task.run(), true);
  test_task.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(bonyuk_p_gauss_gorizont, four_test) {
  int n = 5;
  int m = 5;

  std::vector<int> image = {100, 150, 200, 250, 300, 50,  100, 150, 200, 250, 0,   50,  100, 150,
                            200, 250, 0,   50,  100, 150, 200, 250, 0,   50,  100, 150, 200};

  int min = *std::min_element(image.begin(), image.end());
  int max = *std::max_element(image.begin(), image.end());

  std::vector<int> image_size = {n, m};
  std::vector<int> image_third = {min, max};
  std::vector<int> out(n * m);

  std::vector<int> result = {100, 150, 200, 250, 300, 75, 125, 175, 225, 275, 50, 100, 150, 200, 250,
                             25,  75,  125, 175, 225, 0,  50,  100, 150, 200, 25, 75,  125, 175, 225};

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image_size.data()));
  task_data->inputs_count.emplace_back(image_size.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data->inputs_count.emplace_back(image.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image_third.data()));
  task_data->inputs_count.emplace_back(image_third.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  LinearGaussianFiltering test_task(task_data);
  ASSERT_EQ(test_task.validation(), true);
  test_task.pre_processing();
  ASSERT_EQ(test_task.run(), true);
  test_task.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(bonyuk_p_gauss_gorizont, five_test) {
  int n = 5;
  int m = 5;

  std::vector<int> image = {100, 150, 200, 250, 300,  350, 400, 50,  100,  150, 200, 250, 300,  350, 400,
                            0,   50,  100, 150, +200, 250, 300, 350, 400,  250, 0,   50,  100,  150, 200,
                            250, 300, 350, 400, 0,    50,  100, 150, +200, 250, 300, 350, 400,  250, 0,
                            50,  100, 150, 200, 250,  300, 350, 400, 250,  0,   50,  100, +150, 200, 250,
                            300, 350, 400, 250, 0,    50,  100, 150, 200,  250, 300, 350, 400,  250};

  int min = *std::min_element(image.begin(), image.end());
  int max = *std::max_element(image.begin(), image.end());

  std::vector<int> image_size = {n, m};
  std::vector<int> image_third = {min, max};
  std::vector<int> out(n * m);

  std::vector<int> result = {100, 150, 200, 250, 300, 350, 400, 75,  125, 175, 225, 275, 325, 375, 50,  100,
                             150, 200, 250, 300, 350, 25,  75,  125, 175, 225, 275, 325, 0,   50,  100, 150,
                             200, 250, 300, 0,   25,  75,  125, 175, 225, 275, 0,   0,   25,  75,  125, 175,
                             225, 0,   0,   0,   25,  75,  125, 175, 0,   0,   0,   0,   25,  75,  125, 0,
                             0,   0,   0,   0,   25,  75,  0,   0,   0,   0,   0,   0,   25};

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image_size.data()));
  task_data->inputs_count.emplace_back(image_size.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image.data()));
  task_data->inputs_count.emplace_back(image.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(image_third.data()));
  task_data->inputs_count.emplace_back(image_third.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  LinearGaussianFiltering test_task(task_data);
  ASSERT_EQ(test_task.validation(), true);
  test_task.pre_processing();
  ASSERT_EQ(test_task.run(), true);
  test_task.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(result[i], out[i]);
  }
}
