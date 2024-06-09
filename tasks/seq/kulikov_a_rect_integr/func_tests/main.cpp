// Copyright 2023 Kulikov Artem
#include <gtest/gtest.h>

#include <vector>

#include "seq/kulikov_a_rect_integr/include/ops_seq.hpp"

TEST(kulikov_a_rect_integr_seq, Small) {
  std::vector<double> in{0.0, 1.0, 0.0, 1.0, 1e2};  // x_lim_l, x_lim_u, y_lim_l, y_lim_u, n
  std::vector<double> out(2);                       // res, err

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  KulikovTaskSequential kulikovTaskSequential(taskDataSeq);
  ASSERT_EQ(kulikovTaskSequential.validation(), true);
  kulikovTaskSequential.pre_processing();
  kulikovTaskSequential.run();
  kulikovTaskSequential.post_processing();
  ASSERT_NEAR((double)5 / 6, out[0], out[1]);
}

TEST(kulikov_a_rect_integr_seq, Normal) {
  std::vector<double> in{0.0, 2.0, 3.0, 5.0, 1e3};
  std::vector<double> out(2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  KulikovTaskSequential kulikovTaskSequential(taskDataSeq);
  ASSERT_EQ(kulikovTaskSequential.validation(), true);
  kulikovTaskSequential.pre_processing();
  kulikovTaskSequential.run();
  kulikovTaskSequential.post_processing();
  ASSERT_NEAR((double)64 / 3, out[0], out[1]);
}

TEST(kulikov_a_rect_integr_seq, Zero) {
  std::vector<double> in{0.0, 2.0, 0.0, 0.0, 1e3};
  std::vector<double> out(2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  KulikovTaskSequential kulikovTaskSequential(taskDataSeq);
  ASSERT_EQ(kulikovTaskSequential.validation(), true);
  kulikovTaskSequential.pre_processing();
  kulikovTaskSequential.run();
  kulikovTaskSequential.post_processing();
  ASSERT_NEAR(0.0, out[0], out[1]);
}

TEST(kulikov_a_rect_integr_seq, Negative) {
  std::vector<double> in{-2.0, 0.0, -5.0, -3.0, 1e3};
  std::vector<double> out(2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  KulikovTaskSequential kulikovTaskSequential(taskDataSeq);
  ASSERT_EQ(kulikovTaskSequential.validation(), true);
  kulikovTaskSequential.pre_processing();
  kulikovTaskSequential.run();
  kulikovTaskSequential.post_processing();
  ASSERT_NEAR((double)-32 / 3, out[0], out[1]);
}

TEST(kulikov_a_rect_integr_seq, NegativeLarge) {
  std::vector<double> in{-15.0, 15.0, -20.0, 10.0, 1e3};
  std::vector<double> out(2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  KulikovTaskSequential kulikovTaskSequential(taskDataSeq);
  ASSERT_EQ(kulikovTaskSequential.validation(), true);
  kulikovTaskSequential.pre_processing();
  kulikovTaskSequential.run();
  kulikovTaskSequential.post_processing();
  ASSERT_NEAR((double)63000, out[0], out[1]);
}
