// Copyright 2024 Videneva Ekaterina
#include <gtest/gtest.h>

#include <vector>

#include "tbb/videneva_e_integral/include/ops_tbb.hpp"

TEST(videneva_int_metod_pr_tbb_func, small) {
  std::vector<double> in{0.0, 1.0, 0.0, 1.0, 1e3};

  std::vector<double> out(2);

  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  data_seq->inputs_count.emplace_back(in.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(out.size());

  Videneva_e_tbb_integral::VidenevaEIntegralTBB videneva_opm_task(data_seq);
  ASSERT_EQ(videneva_opm_task.validation(), true);
  videneva_opm_task.pre_processing();
  videneva_opm_task.run();
  videneva_opm_task.post_processing();
  ASSERT_NEAR((double)5 / 6, out[0], out[1]);
}

TEST(videneva_int_metod_pr_tbb_func, normal) {
  std::vector<double> in{0.0, 2.0, 3.0, 5.0, 1e3};
  std::vector<double> out(2);

  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  data_seq->inputs_count.emplace_back(in.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(out.size());

  Videneva_e_tbb_integral::VidenevaEIntegralTBB videneva_opm_task(data_seq);
  ASSERT_EQ(videneva_opm_task.validation(), true);
  videneva_opm_task.pre_processing();
  videneva_opm_task.run();
  videneva_opm_task.post_processing();
  ASSERT_NEAR((double)64 / 3, out[0], out[1]);
}

TEST(videneva_int_metod_pr_tbb_func, null) {
  std::vector<double> in{0.0, 2.0, 0.0, 0.0, 1e3};
  std::vector<double> out(2);

  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  data_seq->inputs_count.emplace_back(in.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(out.size());

  Videneva_e_tbb_integral::VidenevaEIntegralTBB videneva_opm_task(data_seq);
  ASSERT_EQ(videneva_opm_task.validation(), true);
  videneva_opm_task.pre_processing();
  videneva_opm_task.run();
  videneva_opm_task.post_processing();
  ASSERT_NEAR(0.0, out[0], out[1]);
}

TEST(videneva_int_metod_pr_tbb_func, negative) {
  std::vector<double> in{-2.0, 0.0, -5.0, -3.0, 1e3};
  std::vector<double> out(2);

  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  data_seq->inputs_count.emplace_back(in.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(out.size());

  Videneva_e_tbb_integral::VidenevaEIntegralTBB videneva_opm_task(data_seq);
  ASSERT_EQ(videneva_opm_task.validation(), true);
  videneva_opm_task.pre_processing();
  videneva_opm_task.run();
  videneva_opm_task.post_processing();
  ASSERT_NEAR((double)-32 / 3, out[0], out[1]);
}

TEST(videneva_int_metod_pr_tbb_func, negativeLarge) {
  std::vector<double> in{-15.0, 15.0, -20.0, 10.0, 1e3};
  std::vector<double> out(2);

  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  data_seq->inputs_count.emplace_back(in.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(out.size());

  Videneva_e_tbb_integral::VidenevaEIntegralTBB videneva_opm_task(data_seq);
  ASSERT_EQ(videneva_opm_task.validation(), true);
  videneva_opm_task.pre_processing();
  videneva_opm_task.run();
  videneva_opm_task.post_processing();
  ASSERT_NEAR((double)63000, out[0], out[1]);
}
