// Copyright 2024 Kulagin Aleksandr
#include <gtest/gtest.h>

#include "seq/kulagin_a_gauss_filter_vert/include/ops_seq.hpp"

void test_case(const std::string &filename, float sigma, const std::string &filename_res) {
  // Create data
  bool is_good;
  size_t w;
  size_t h;
  size_t _w;
  size_t _h;
  std::vector<uint32_t> img(1);
  std::vector<uint32_t> res(1);
  is_good = kulagin_a_gauss::get_img_from_file(
      kulagin_a_gauss::generate_path_to_img("seq", "kulagin_a_gauss_filter_vert", filename), img, w, h);
  ASSERT_EQ(is_good, true);
  is_good = kulagin_a_gauss::get_img_from_file(
      kulagin_a_gauss::generate_path_to_img("seq", "kulagin_a_gauss_filter_vert", filename_res), res, _w, _h);
  ASSERT_EQ(is_good, true);
  ASSERT_EQ(_w, w);
  ASSERT_EQ(_h, h);
  std::vector<float> kernel = kulagin_a_gauss::generate_kernel(sigma);
  std::vector<uint32_t> out(w * h);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(kernel.data()));
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(w);
  taskDataSeq->outputs_count.emplace_back(h);

  // Create Task
  FilterGaussVerticalTaskSequential myTask(taskDataSeq);
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

TEST(kulagin_a_gauss_filter_vert_seq, test1) { test_case("test1.dat", 0.0f, "test1_res.dat"); }

TEST(kulagin_a_gauss_filter_vert_seq, test2) { test_case("test2.dat", 0.0f, "test2_res.dat"); }

TEST(kulagin_a_gauss_filter_vert_seq, test3) { test_case("test3.dat", 0.0f, "test3_res.dat"); }

TEST(kulagin_a_gauss_filter_vert_seq, test4) { test_case("test4.dat", 2.0f, "test4_res.dat"); }

TEST(kulagin_a_gauss_filter_vert_seq, test5) { test_case("test5.dat", 0.0f, "test5_res.dat"); }
