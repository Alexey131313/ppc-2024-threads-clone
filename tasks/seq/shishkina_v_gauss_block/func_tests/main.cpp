// Copyright 2023 Shishkina Valeriya
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "seq/shishkina_v_gauss_block/include/ops_seq.hpp"

TEST(shishkina_v_gauss_block, test1) {
  int n = 3;
  int m = 3;

  // Create data
  std::vector<int> tmp = {255, 255, 255, 255, 255, 255, 255, 255, 255};

  int min = *std::min_element(tmp.begin(), tmp.end());
  int max = *std::max_element(tmp.begin(), tmp.end());

  std::vector<int> tmp_sec = {n, m};
  std::vector<int> tmp_third = {min, max};
  std::vector<int> out(n * m);

  std::vector<int> res = {255, 255, 255, 255, 255, 255, 255, 255, 255};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_sec.data()));
  taskDataSeq->inputs_count.emplace_back(tmp_sec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  taskDataSeq->inputs_count.emplace_back(tmp.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_third.data()));
  taskDataSeq->inputs_count.emplace_back(tmp_third.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  LinearFilteringGauss testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  ASSERT_EQ(testTaskSequential.run(), true);
  testTaskSequential.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(res[i], out[i]);
  }
}

TEST(shishkina_v_gauss_block, test2) {
  int n = 3;
  int m = 3;

  // Create data
  std::vector<int> tmp = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  int min = *std::min_element(tmp.begin(), tmp.end());
  int max = *std::max_element(tmp.begin(), tmp.end());

  std::vector<int> tmp_sec = {n, m};
  std::vector<int> tmp_third = {min, max};
  std::vector<int> out(n * m);

  std::vector<int> res = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_sec.data()));
  taskDataSeq->inputs_count.emplace_back(tmp_sec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  taskDataSeq->inputs_count.emplace_back(tmp.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_third.data()));
  taskDataSeq->inputs_count.emplace_back(tmp_third.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  LinearFilteringGauss testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  ASSERT_EQ(testTaskSequential.run(), true);
  testTaskSequential.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(res[i], out[i]);
  }
}

TEST(shishkina_v_gauss_block, test3) {
  int n = 4;
  int m = 4;

  // Create data
  std::vector<int> tmp = {0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0};

  int min = *std::min_element(tmp.begin(), tmp.end());
  int max = *std::max_element(tmp.begin(), tmp.end());

  std::vector<int> tmp_sec = {n, m};
  std::vector<int> tmp_third = {min, max};
  std::vector<int> out(n * m);

  std::vector<int> res = {4, 6, 6, 5, 6, 8, 8, 6, 6, 8, 8, 6, 5, 6, 6, 4};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_sec.data()));
  taskDataSeq->inputs_count.emplace_back(tmp_sec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  taskDataSeq->inputs_count.emplace_back(tmp.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_third.data()));
  taskDataSeq->inputs_count.emplace_back(tmp_third.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  LinearFilteringGauss testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  ASSERT_EQ(testTaskSequential.run(), true);
  testTaskSequential.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(res[i], out[i]);
  }
}

TEST(shishkina_v_gauss_block, test4) {
  int n = 5;
  int m = 5;

  // Create data
  std::vector<int> tmp = {0,   255, 0,   255, 0,   255, 0,   255, 0,   255, 0,   255, 0,
                          255, 0,   255, 0,   255, 0,   255, 0,   255, 0,   255, 0};

  int min = *std::min_element(tmp.begin(), tmp.end());
  int max = *std::max_element(tmp.begin(), tmp.end());

  std::vector<int> tmp_sec = {n, m};
  std::vector<int> tmp_third = {min, max};
  std::vector<int> out(n * m);

  std::vector<int> res = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                          255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_sec.data()));
  taskDataSeq->inputs_count.emplace_back(tmp_sec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  taskDataSeq->inputs_count.emplace_back(tmp.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_third.data()));
  taskDataSeq->inputs_count.emplace_back(tmp_third.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  LinearFilteringGauss testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  ASSERT_EQ(testTaskSequential.run(), true);
  testTaskSequential.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(res[i], out[i]);
  }
}

TEST(shishkina_v_gauss_block, test5) {
  int n = 3;
  int m = 5;

  // Create data
  std::vector<int> tmp = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};

  int min = *std::min_element(tmp.begin(), tmp.end());
  int max = *std::max_element(tmp.begin(), tmp.end());

  std::vector<int> tmp_sec = {n, m};
  std::vector<int> tmp_third = {min, max};
  std::vector<int> out(n * m);

  std::vector<int> res = {5, 6, 6, 6, 5, 6, 8, 8, 8, 6, 5, 6, 6, 6, 5};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_sec.data()));
  taskDataSeq->inputs_count.emplace_back(tmp_sec.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  taskDataSeq->inputs_count.emplace_back(tmp.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp_third.data()));
  taskDataSeq->inputs_count.emplace_back(tmp_third.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  LinearFilteringGauss testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  ASSERT_EQ(testTaskSequential.run(), true);
  testTaskSequential.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(res[i], out[i]);
  }
}
