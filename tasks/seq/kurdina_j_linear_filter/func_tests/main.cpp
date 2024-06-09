// Copyright 2024 Kurdina Julia
#include <gtest/gtest.h>

#include <vector>

#include "seq/kurdina_j_linear_filter/include/ops_seq.hpp"

TEST(kurdina_j_linear_filter_seq, Image_1) {
  int n = 3;
  int m = 3;
  std::vector<uint8_t> gauss_kernel = {1, 1, 1, 1, 9, 1, 1, 1, 1};

  // Create data
  std::vector<uint8_t> in = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> size_m = {n, m};
  std::vector<uint8_t> out(n * m);
  std::vector<uint8_t> result = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(size_m.data()));
  taskDataSeq->inputs_count.emplace_back(size_m.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(gauss_kernel.data()));
  taskDataSeq->inputs_count.emplace_back(gauss_kernel.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  HorizontalSplitSequential horizontalsplitSequential(taskDataSeq);
  ASSERT_EQ(horizontalsplitSequential.validation(), true);
  horizontalsplitSequential.pre_processing();
  horizontalsplitSequential.run();
  horizontalsplitSequential.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(kurdina_j_linear_filter_seq, Image_2) {
  int n = 3;
  int m = 3;
  std::vector<uint8_t> gauss_kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  // Create data
  std::vector<uint8_t> in = {255, 255, 255, 255, 255, 255, 255, 255, 255};
  std::vector<int> size_m = {n, m};
  std::vector<uint8_t> out(n * m);
  std::vector<uint8_t> result = {255, 255, 255, 255, 255, 255, 255, 255, 255};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(size_m.data()));
  taskDataSeq->inputs_count.emplace_back(size_m.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(gauss_kernel.data()));
  taskDataSeq->inputs_count.emplace_back(gauss_kernel.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  HorizontalSplitSequential horizontalsplitSequential(taskDataSeq);
  ASSERT_EQ(horizontalsplitSequential.validation(), true);
  horizontalsplitSequential.pre_processing();
  horizontalsplitSequential.run();
  horizontalsplitSequential.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(kurdina_j_linear_filter_seq, Image_3) {
  int n = 3;
  int m = 3;
  std::vector<uint8_t> gauss_kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  // Create data
  std::vector<uint8_t> in = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> size_m = {n, m};
  std::vector<uint8_t> out(n * m);
  std::vector<uint8_t> result = {5, 13, 31, 26, 78, 155, 134, 255, 255};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(size_m.data()));
  taskDataSeq->inputs_count.emplace_back(size_m.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(gauss_kernel.data()));
  taskDataSeq->inputs_count.emplace_back(gauss_kernel.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  HorizontalSplitSequential horizontalsplitSequential(taskDataSeq);
  ASSERT_EQ(horizontalsplitSequential.validation(), true);
  horizontalsplitSequential.pre_processing();
  horizontalsplitSequential.run();
  horizontalsplitSequential.post_processing();
  for (int i = 0; i < n * m; i++) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(kurdina_j_linear_filter_seq, Check_validation_1) {
  int n = 3;
  int m = 3;
  std::vector<uint8_t> gauss_kernel(20);

  // Create data
  std::vector<uint8_t> in = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> size_m = {n, m};
  std::vector<uint8_t> out(n * m);
  std::vector<uint8_t> result = {5, 13, 31, 26, 78, 155, 134, 255, 255};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(size_m.data()));
  taskDataSeq->inputs_count.emplace_back(size_m.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(gauss_kernel.data()));
  taskDataSeq->inputs_count.emplace_back(gauss_kernel.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  HorizontalSplitSequential horizontalsplitSequential(taskDataSeq);
  ASSERT_EQ(horizontalsplitSequential.validation(), false);
  horizontalsplitSequential.pre_processing();
  horizontalsplitSequential.run();
  horizontalsplitSequential.post_processing();
}

TEST(kurdina_j_linear_filter_seq, Check_validation_2) {
  int n = 3;
  int m = 3;
  std::vector<uint8_t> gauss_kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  // Create data
  std::vector<uint8_t> in = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> size_m = {n, m};
  std::vector<uint8_t> out(n * m * 100);
  std::vector<uint8_t> result = {5, 13, 31, 26, 78, 155, 134, 255, 255};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(size_m.data()));
  taskDataSeq->inputs_count.emplace_back(size_m.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(gauss_kernel.data()));
  taskDataSeq->inputs_count.emplace_back(gauss_kernel.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  HorizontalSplitSequential horizontalsplitSequential(taskDataSeq);
  ASSERT_EQ(horizontalsplitSequential.validation(), false);
  horizontalsplitSequential.pre_processing();
  horizontalsplitSequential.run();
  horizontalsplitSequential.post_processing();
}