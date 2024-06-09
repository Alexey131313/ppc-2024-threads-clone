// Copyright 2024 Kulagin Aleksandr
#include <gtest/gtest.h>
#include <omp.h>

#include "omp/kulagin_a_gauss_filter_vert/include/ops_omp.hpp"

void test_case(const size_t& w, const size_t& h, const float& sigma,
               std::vector<uint32_t> (*generator)(const size_t&, const size_t&)) {
  // Create data
  std::vector<uint32_t> img = generator(w, h);
  std::vector<uint32_t> res(w * h);
  std::vector<float> kernel = kulagin_a_gauss::generate_kernel(sigma);
  std::vector<uint32_t> out(w * h);
  kulagin_a_gauss::apply_filter(w, h, img.data(), kernel.data(), res.data());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(w);
  taskDataSeq->outputs_count.emplace_back(h);

  // Create Task
  FilterGaussVerticalTaskOMPKulagin myTask(taskDataSeq);
  ASSERT_EQ(myTask.validation(), true);
  ASSERT_EQ(myTask.pre_processing(), true);
  ASSERT_EQ(myTask.run(), true);
  ASSERT_EQ(myTask.post_processing(), true);
  for (size_t i = 0; i < w * h; i++) {
    for (size_t j = 0; j < 3; j++) {
      uint32_t res_ch = kulagin_a_gauss::get_color_channel(res[i], j);
      uint32_t out_ch = kulagin_a_gauss::get_color_channel(out[i], j);
      ASSERT_EQ(((res_ch >= out_ch) ? (res_ch - out_ch) : (out_ch - res_ch)) <= 1, true);
    }
  }
}

TEST(kulagin_a_gauss_filter_vert_omp, test1) { test_case(100, 100, 0.0f, kulagin_a_gauss::generator1); }

TEST(kulagin_a_gauss_filter_vert_omp, test2) { test_case(100, 200, 0.0f, kulagin_a_gauss::generator1); }

TEST(kulagin_a_gauss_filter_vert_omp, test3) { test_case(200, 100, 0.0f, kulagin_a_gauss::generator1); }

TEST(kulagin_a_gauss_filter_vert_omp, test4) { test_case(100, 100, 4.0f, kulagin_a_gauss::generator1); }

TEST(kulagin_a_gauss_filter_vert_omp, test5) { test_case(101, 101, 2.0f, kulagin_a_gauss::generator1); }
