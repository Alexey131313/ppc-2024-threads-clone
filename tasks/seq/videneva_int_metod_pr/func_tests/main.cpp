// Copyright 2023 Videneva Ekaterina
#include <gtest/gtest.h>

#include <vector>

#include "seq/videneva_int_metod_pr/include/ops_seq.hpp"

TEST(videneva_e_int_metod_pr_func_tests, small) {
  std::vector<double> in{0.0, 1.0, 0.0, 1.0, 1e2};
  std::vector<double> out(2);

  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  data_seq->inputs_count.emplace_back(in.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(out.size());

  Videneva_seq_task videneva_seq_task(data_seq);
  ASSERT_EQ(videneva_seq_task.validation(), true);
  videneva_seq_task.pre_processing();
  videneva_seq_task.run();
  videneva_seq_task.post_processing();
  ASSERT_NEAR((double)5 / 6, out[0], out[1]);
}

TEST(videneva_e_int_metod_pr_func_tests, normal) {
  std::vector<double> in{0.0, 2.0, 3.0, 5.0, 1e3};
  std::vector<double> out(2);

  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  data_seq->inputs_count.emplace_back(in.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(out.size());

  Videneva_seq_task videneva_seq_task(data_seq);
  ASSERT_EQ(videneva_seq_task.validation(), true);
  videneva_seq_task.pre_processing();
  videneva_seq_task.run();
  videneva_seq_task.post_processing();
  ASSERT_NEAR((double)64 / 3, out[0], out[1]);
}

TEST(videneva_e_int_metod_pr_func_tests, null) {
  std::vector<double> in{0.0, 2.0, 0.0, 0.0, 1e3};
  std::vector<double> out(2);

  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  data_seq->inputs_count.emplace_back(in.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(out.size());

  Videneva_seq_task videneva_seq_task(data_seq);
  ASSERT_EQ(videneva_seq_task.validation(), true);
  videneva_seq_task.pre_processing();
  videneva_seq_task.run();
  videneva_seq_task.post_processing();
  ASSERT_NEAR(0.0, out[0], out[1]);
}

TEST(videneva_e_int_metod_pr_func_tests, negative) {
  std::vector<double> in{-2.0, 0.0, -5.0, -3.0, 1e3};
  std::vector<double> out(2);

  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  data_seq->inputs_count.emplace_back(in.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(out.size());

  Videneva_seq_task videneva_seq_task(data_seq);
  ASSERT_EQ(videneva_seq_task.validation(), true);
  videneva_seq_task.pre_processing();
  videneva_seq_task.run();
  videneva_seq_task.post_processing();
  ASSERT_NEAR((double)-32 / 3, out[0], out[1]);
}

TEST(videneva_e_int_metod_pr_func_tests, NegativeLarge) {
  std::vector<double> in{-15.0, 15.0, -20.0, 10.0, 1e3};
  std::vector<double> out(2);

  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  data_seq->inputs_count.emplace_back(in.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(out.size());

  Videneva_seq_task videneva_seq_task(data_seq);
  ASSERT_EQ(videneva_seq_task.validation(), true);
  videneva_seq_task.pre_processing();
  videneva_seq_task.run();
  videneva_seq_task.post_processing();
  ASSERT_NEAR((double)63000, out[0], out[1]);
}