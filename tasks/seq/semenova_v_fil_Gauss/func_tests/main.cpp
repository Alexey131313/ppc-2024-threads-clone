// Copyright 2024 Semenova Veronika
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/semenova_v_fil_Gauss/include/ops_seq.hpp"

void CreateRandomVector(int *vec, int n) {
  // std::srand(0);
  for (int i = 0; i < n; i++) {
    vec[i] = std::rand() % 256;
  }
}

TEST(semenova_v_fil_Gauss_seq, Creat_empty_task) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_FALSE(ImageFilGauss.validation());
}
TEST(semenova_v_fil_Gauss_seq, Creat_task_with_empty_outputs_and_inputs) {
  int n = 3;
  int m = 3;
  int *image = new int[n * m];
  int *filteredImage = new int[n * m];

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(filteredImage));
  taskDataSeq->outputs_count.emplace_back(n);
  taskDataSeq->outputs_count.emplace_back(m);

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_TRUE(ImageFilGauss.validation());
  delete[] image;
  delete[] filteredImage;
}
TEST(semenova_v_fil_Gauss_seq, Creat_task_without_inputs1) {
  int n = 3;
  int m = 3;
  int *filteredImage = new int[n * m];

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(filteredImage));
  taskDataSeq->outputs_count.emplace_back(n);
  taskDataSeq->outputs_count.emplace_back(m);

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_FALSE(ImageFilGauss.validation());
  delete[] filteredImage;
}
TEST(semenova_v_fil_Gauss_seq, Creat_task_without_inputs2) {
  int n = 3;
  int m = 3;
  int *image = new int[n * m];
  int *filteredImage = new int[n * m];

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(filteredImage));
  taskDataSeq->outputs_count.emplace_back(n);
  taskDataSeq->outputs_count.emplace_back(m);

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_FALSE(ImageFilGauss.validation());
  delete[] image;
  delete[] filteredImage;
}
TEST(semenova_v_fil_Gauss_seq, Creat_task_without_inputs3) {
  int n = 3;
  int m = 3;
  int *filteredImage = new int[n * m];

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(filteredImage));
  taskDataSeq->outputs_count.emplace_back(n);
  taskDataSeq->outputs_count.emplace_back(m);

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_FALSE(ImageFilGauss.validation());
  delete[] filteredImage;
}
TEST(semenova_v_fil_Gauss_seq, Creat_task_without_outputs1) {
  int n = 3;
  int m = 3;
  int *image = new int[n * m];

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_FALSE(ImageFilGauss.validation());
  delete[] image;
}
TEST(semenova_v_fil_Gauss_seq, Creat_task_without_outputs2) {
  int n = 3;
  int m = 3;
  int *image = new int[n * m];
  int *filteredImage = new int[n * m];

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(filteredImage));

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_FALSE(ImageFilGauss.validation());
  delete[] image;
  delete[] filteredImage;
}
TEST(semenova_v_fil_Gauss_seq, Creat_task_without_outputs3) {
  int n = 3;
  int m = 3;
  int *image = new int[n * m];

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  taskDataSeq->outputs_count.emplace_back(n);
  taskDataSeq->outputs_count.emplace_back(m);

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_FALSE(ImageFilGauss.validation());
  delete[] image;
}
TEST(semenova_v_fil_Gauss_seq, Creat_task_with_wrong_matrix_size1) {
  int n = 3;
  int m = 3;
  int a = 6;
  int b = 6;
  int *image = new int[n * m];
  int *filteredImage = new int[n * m];

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(filteredImage));
  taskDataSeq->outputs_count.emplace_back(a);
  taskDataSeq->outputs_count.emplace_back(b);

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_FALSE(ImageFilGauss.validation());
  delete[] image;
  delete[] filteredImage;
}
TEST(semenova_v_fil_Gauss_seq, Creat_task_with_wrong_matrix_size2) {
  int n = 2;
  int m = 2;
  int *image = new int[n * m];
  int *filteredImage = new int[n * m];

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(filteredImage));
  taskDataSeq->outputs_count.emplace_back(n);
  taskDataSeq->outputs_count.emplace_back(m);

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_FALSE(ImageFilGauss.validation());
  delete[] image;
  delete[] filteredImage;
}
TEST(semenova_v_fil_Gauss_seq, Creat_task_with_outputs_and_inputs2) {
  int n = 3;
  int m = 3;
  int *filteredImage = new int[n * m];
  int image[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(filteredImage));
  taskDataSeq->outputs_count.emplace_back(n);
  taskDataSeq->outputs_count.emplace_back(m);

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_TRUE(ImageFilGauss.validation());
  ASSERT_TRUE(ImageFilGauss.pre_processing());
  delete[] filteredImage;
}
TEST(semenova_v_fil_Gauss_seq, Run_task2) {
  int n = 3;
  int m = 3;
  int *filteredImage = new int[n * m];
  int image[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(filteredImage));
  taskDataSeq->outputs_count.emplace_back(n);
  taskDataSeq->outputs_count.emplace_back(m);

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_TRUE(ImageFilGauss.validation());
  ImageFilGauss.pre_processing();
  ASSERT_TRUE(ImageFilGauss.run());
  delete[] filteredImage;
}
TEST(semenova_v_fil_Gauss_seq, Task_post_processing2) {
  int n = 3;
  int m = 3;
  int *filteredImage = new int[n * m];
  int image[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(filteredImage));
  taskDataSeq->outputs_count.emplace_back(n);
  taskDataSeq->outputs_count.emplace_back(m);

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_TRUE(ImageFilGauss.validation());
  ImageFilGauss.pre_processing();
  ImageFilGauss.run();
  ASSERT_TRUE(ImageFilGauss.post_processing());
  delete[] filteredImage;
}
TEST(semenova_v_fil_Gauss_seq, Task_run_correct1) {
  int n = 3;
  int m = 3;
  int *filteredImage = new int[n * m];
  int image[9] = {1, 50, 3, 4, 4, 0, 7, 8, 98};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(filteredImage));
  taskDataSeq->outputs_count.emplace_back(n);
  taskDataSeq->outputs_count.emplace_back(m);

  int res[9] = {1, 50, 3, 4, 15, 0, 7, 8, 98};

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_TRUE(ImageFilGauss.validation());
  ImageFilGauss.pre_processing();
  ImageFilGauss.run();
  ImageFilGauss.post_processing();

  for (int i = 0; i < n * m; ++i) {
    EXPECT_EQ(filteredImage[i], res[i]);
  }
  delete[] filteredImage;
}
TEST(semenova_v_fil_Gauss_seq, Task_run_correct2) {
  int n = 4;
  int m = 4;
  int *filteredImage = new int[n * m];
  int image[16] = {1, 50, 3, 125, 1, 4, 0, 69, 0, 7, 8, 98, 0, 0, 0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(filteredImage));
  taskDataSeq->outputs_count.emplace_back(n);
  taskDataSeq->outputs_count.emplace_back(m);

  int res[16] = {1, 50, 3, 125, 1, 9, 28, 69, 0, 3, 19, 98, 0, 0, 0, 0};

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_TRUE(ImageFilGauss.validation());
  ImageFilGauss.pre_processing();
  ImageFilGauss.run();
  ImageFilGauss.post_processing();

  for (int i = 0; i < n * m; ++i) {
    EXPECT_EQ(filteredImage[i], res[i]);
  }
  delete[] filteredImage;
}
TEST(semenova_v_fil_Gauss_seq, Task_correct_pre_processing) {
  int n = 4;
  int m = 4;
  int *filteredImage = new int[n * m];
  int image[16] = {285, 285, 285, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(filteredImage));
  taskDataSeq->outputs_count.emplace_back(n);
  taskDataSeq->outputs_count.emplace_back(m);

  int res[16] = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_TRUE(ImageFilGauss.validation());
  ImageFilGauss.pre_processing();
  ImageFilGauss.run();
  ImageFilGauss.post_processing();

  for (int i = 0; i < n * m; ++i) {
    EXPECT_EQ(filteredImage[i], res[i]);
  }

  delete[] filteredImage;
}
TEST(semenova_v_fil_Gauss_seq, Task_correct_with_random_image) {
  int n = 10;
  int m = 10;
  int *image = new int[n * m];
  int *filteredImage = new int[n * m];
  CreateRandomVector(image, n * m);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(image));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(filteredImage));
  taskDataSeq->outputs_count.emplace_back(n);
  taskDataSeq->outputs_count.emplace_back(m);

  ImageFilGauss ImageFilGauss(taskDataSeq);
  ASSERT_TRUE(ImageFilGauss.validation());
  ImageFilGauss.pre_processing();
  ImageFilGauss.run();
  ASSERT_TRUE(ImageFilGauss.post_processing());
  delete[] image;
  delete[] filteredImage;
}
