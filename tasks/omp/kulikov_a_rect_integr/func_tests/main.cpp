// Copyright 2023 Kulikov Artem
#include <gtest/gtest.h>

#include <vector>

#include "omp/kulikov_a_rect_integr/include/ops_omp.hpp"

TEST(kulikov_a_rect_integr_omp, Small) {
  std::vector<double> in{0.0, 1.0, 0.0, 1.0, 1e3};  // x_lim_l, x_lim_u, y_lim_l, y_lim_u, n
  std::vector<double> out(2);                       // res, err

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  KulikovTaskOMP kulikovTaskOMP(taskDataSeq);
  ASSERT_EQ(kulikovTaskOMP.validation(), true);
  kulikovTaskOMP.pre_processing();
  kulikovTaskOMP.run();
  kulikovTaskOMP.post_processing();
  ASSERT_NEAR((double)5 / 6, out[0], out[1]);
}

TEST(kulikov_a_rect_integr_omp, Normal) {
  std::vector<double> in{0.0, 2.0, 3.0, 5.0, 1e3};
  std::vector<double> out(2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  KulikovTaskOMP kulikovTaskOMP(taskDataSeq);
  ASSERT_EQ(kulikovTaskOMP.validation(), true);
  kulikovTaskOMP.pre_processing();
  kulikovTaskOMP.run();
  kulikovTaskOMP.post_processing();
  ASSERT_NEAR((double)64 / 3, out[0], out[1]);
}

TEST(kulikov_a_rect_integr_omp, Zero) {
  std::vector<double> in{0.0, 2.0, 0.0, 0.0, 1e3};
  std::vector<double> out(2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  KulikovTaskOMP kulikovTaskOMP(taskDataSeq);
  ASSERT_EQ(kulikovTaskOMP.validation(), true);
  kulikovTaskOMP.pre_processing();
  kulikovTaskOMP.run();
  kulikovTaskOMP.post_processing();
  ASSERT_NEAR(0.0, out[0], out[1]);
}

TEST(kulikov_a_rect_integr_omp, Negative) {
  std::vector<double> in{-2.0, 0.0, -5.0, -3.0, 1e3};
  std::vector<double> out(2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  KulikovTaskOMP kulikovTaskOMP(taskDataSeq);
  ASSERT_EQ(kulikovTaskOMP.validation(), true);
  kulikovTaskOMP.pre_processing();
  kulikovTaskOMP.run();
  kulikovTaskOMP.post_processing();
  ASSERT_NEAR((double)-32 / 3, out[0], out[1]);
}

TEST(kulikov_a_rect_integr_omp, NegativeLarge) {
  std::vector<double> in{-15.0, 15.0, -20.0, 10.0, 1e3};
  std::vector<double> out(2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  KulikovTaskOMP kulikovTaskOMP(taskDataSeq);
  ASSERT_EQ(kulikovTaskOMP.validation(), true);
  kulikovTaskOMP.pre_processing();
  kulikovTaskOMP.run();
  kulikovTaskOMP.post_processing();
  ASSERT_NEAR((double)63000, out[0], out[1]);
}