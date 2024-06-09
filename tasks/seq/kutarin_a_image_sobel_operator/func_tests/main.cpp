// Copyright 2024 Kutarin Alexander
#include <gtest/gtest.h>

#include <fstream>
#include <random>
#include <vector>

#include "seq/kutarin_a_image_sobel_operator/include/ops_seq.hpp"

TEST(kutarin_a_image_sobel_operator, Test_Sobel_Simple_Image_5x5) {
  const int height = 5;
  const int width = 5;

  std::vector<int> inImage = {142, 249, 236, 158, 98, 238, 32,  196, 59,  75,  9,   188, 12,
                              213, 145, 178, 140, 65, 174, 156, 239, 196, 125, 152, 122};

  std::vector<int> outImage(height * width, 0);

  std::vector<int> resultImage = {135, 255, 255, 255, 234, 255, 255, 255, 255, 218, 134, 151, 117,
                                  255, 255, 255, 255, 179, 255, 186, 255, 255, 182, 83,  164};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inImage.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outImage.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  SobelTaskSequential sobelTaskSequential(taskDataSeq);
  ASSERT_EQ(sobelTaskSequential.validation(), true);
  sobelTaskSequential.pre_processing();
  sobelTaskSequential.run();
  sobelTaskSequential.post_processing();

  for (int i = 0; i < width * height; ++i) {
    ASSERT_EQ(resultImage[i], outImage[i]);
  }
}

TEST(kutarin_a_image_sobel_operator, Test_Sobel_Simple_Image_5x7) {
  const int width = 5;
  const int height = 7;

  std::vector<int> inImage = {103, 93,  44,  126, 174, 31, 163, 206, 164, 183, 150, 112, 87,  117, 59, 0, 52, 193,
                              22,  186, 234, 179, 56,  76, 177, 97,  62,  249, 80,  130, 135, 145, 1,  7, 149};

  std::vector<int> outImage(height * width, 0);

  std::vector<int> resultImage = {178, 230, 255, 255, 175, 255, 255, 104, 106, 255, 230, 255,
                                  255, 255, 148, 255, 236, 162, 79,  255, 255, 173, 255, 164,
                                  255, 255, 222, 255, 223, 255, 197, 253, 255, 255, 255};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inImage.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outImage.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  SobelTaskSequential sobelTaskSequential(taskDataSeq);
  ASSERT_EQ(sobelTaskSequential.validation(), true);
  sobelTaskSequential.pre_processing();
  sobelTaskSequential.run();
  sobelTaskSequential.post_processing();

  for (int i = 0; i < width * height; ++i) {
    ASSERT_EQ(resultImage[i], outImage[i]);
  }
}

TEST(kutarin_a_image_sobel_operator, Test_Sobel_Simple_Image_11x11) {
  const int height = 11;
  const int width = 11;

  std::vector<int> inImage = {255, 255, 80,  255, 80,  80,  80, 255, 80,  80,  255, 80,  80, 80, 255, 80,  80,  80,
                              255, 80,  80,  80,  80,  80,  80, 255, 80,  80,  80,  255, 80, 80, 80,  80,  80,  80,
                              255, 80,  80,  80,  255, 80,  80, 80,  80,  80,  80,  255, 80, 80, 80,  255, 80,  80,
                              80,  80,  80,  80,  255, 80,  80, 80,  255, 80,  80,  80,  80, 80, 80,  255, 80,  80,
                              80,  255, 80,  80,  80,  255, 80, 80,  80,  80,  80,  80,  80, 80, 80,  80,  255, 80,
                              80,  80,  255, 80,  80,  80,  80, 80,  255, 255, 255, 80,  80, 80, 80,  255, 255, 255,
                              80,  255, 255, 255, 255, 80,  80, 80,  80,  80,  80,  255, 255};

  std::vector<int> outImage(height * width, 0);

  std::vector<int> resultImage = {
      255, 255, 247, 0,   255, 0,   255, 0,   255, 255, 255, 255, 255, 255, 0,   255, 0,   255, 0,   255, 247,
      255, 0,   0,   255, 0,   255, 0,   255, 0,   255, 0,   0,   0,   0,   255, 0,   255, 0,   255, 0,   255,
      0,   0,   0,   0,   255, 0,   255, 0,   255, 0,   255, 0,   0,   0,   0,   255, 0,   255, 0,   255, 0,
      255, 0,   0,   255, 247, 255, 255, 255, 0,   255, 255, 255, 0,   0,   255, 255, 247, 247, 247, 247, 247,
      255, 247, 247, 255, 255, 255, 247, 255, 0,   247, 255, 255, 255, 255, 255, 247, 255, 255, 0,   255, 247,
      255, 0,   247, 255, 255, 0,   247, 255, 255, 0,   247, 255, 255, 255, 255, 247,
  };

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inImage.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outImage.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  SobelTaskSequential sobelTaskSequential(taskDataSeq);
  ASSERT_EQ(sobelTaskSequential.validation(), true);
  sobelTaskSequential.pre_processing();
  sobelTaskSequential.run();
  sobelTaskSequential.post_processing();

  for (int i = 0; i < width * height; ++i) {
    ASSERT_EQ(resultImage[i], outImage[i]);
  }
}

TEST(kutarin_a_image_sobel_operator, Test_Sobel_White_Image) {
  const int height = 5;
  const int width = 5;

  std::vector<int> inImage = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

  std::vector<int> outImage(height * width, 0);

  std::vector<int> resultImage = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inImage.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outImage.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  SobelTaskSequential sobelTaskSequential(taskDataSeq);
  ASSERT_EQ(sobelTaskSequential.validation(), true);
  sobelTaskSequential.pre_processing();
  sobelTaskSequential.run();
  sobelTaskSequential.post_processing();

  for (int i = 0; i < width * height; ++i) {
    ASSERT_EQ(resultImage[i], outImage[i]);
  }
}

TEST(kutarin_a_image_sobel_operator, Test_Sobel_Black_Image) {
  const int width = 5;
  const int height = 5;

  std::vector<int> inImage = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int> outImage(height * width, 0);

  std::vector<int> resultImage = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inImage.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outImage.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  SobelTaskSequential sobelTaskSequential(taskDataSeq);
  ASSERT_EQ(sobelTaskSequential.validation(), true);
  sobelTaskSequential.pre_processing();
  sobelTaskSequential.run();
  sobelTaskSequential.post_processing();

  for (int i = 0; i < width * height; ++i) {
    ASSERT_EQ(resultImage[i], outImage[i]);
  }
}

TEST(kutarin_a_image_sobel_operator, Test_Sobel_Random_Image) {
  const int width = 256;
  const int height = 256;

  std::vector<int> inImage(width * height, 0);

  std::default_random_engine rnd;
  std::uniform_int_distribution<> dist(0, 255);

  for (int i = 0; i < width * height; ++i) {
    inImage[i] = dist(rnd);
  }

  std::vector<int> outImage(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inImage.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outImage.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  SobelTaskSequential sobelTaskSequential(taskDataSeq);
  ASSERT_EQ(sobelTaskSequential.validation(), true);
  sobelTaskSequential.pre_processing();
  sobelTaskSequential.run();
  sobelTaskSequential.post_processing();
}

TEST(kutarin_a_image_sobel_operator, Test_Sobel_Random_Image_2) {
  const int width = 300;
  const int height = 128;

  std::vector<int> inImage(width * height, 0);

  std::default_random_engine rnd;
  std::uniform_int_distribution<> dist(0, 255);

  for (int i = 0; i < width * height; ++i) {
    inImage[i] = dist(rnd);
  }

  std::vector<int> outImage(width * height, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inImage.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outImage.data()));
  taskDataSeq->outputs_count.emplace_back(width);
  taskDataSeq->outputs_count.emplace_back(height);

  SobelTaskSequential sobelTaskSequential(taskDataSeq);
  ASSERT_EQ(sobelTaskSequential.validation(), true);
  sobelTaskSequential.pre_processing();
  sobelTaskSequential.run();
  sobelTaskSequential.post_processing();
}
