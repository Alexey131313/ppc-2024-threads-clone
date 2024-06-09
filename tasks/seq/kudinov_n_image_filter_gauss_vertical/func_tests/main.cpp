// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/kudinov_n_image_filter_gauss_vertical/include/image_filter_gauss_vertical_task.hpp"

TEST(kudinov_n_image_filter_gauss_vertical_seq, validation_error) {
  // Create data
  std::size_t radius = 1;
  double sigma = -23.3;
  auto gauss_kernel = GaussKernel(radius, sigma);

  std::size_t height = 5;
  std::size_t width = 3;
  // clang-format off
  std::vector<Pixel> pixels = {
    11, 22, 33,
    12, 23, 34,
  };
  // clang-format on
  auto input_image = Image(height, width, pixels);
  auto output_image = Image(height, width, pixels);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&gauss_kernel));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_image));
  taskData->inputs_count.emplace_back(1);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&output_image));
  taskData->outputs_count.emplace_back(1);

  // Create Task
  ImageFilterGaussVerticalTask task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(kudinov_n_image_filter_gauss_vertical_seq, square_small_image) {
  // Create data
  std::size_t radius = 1;
  double sigma = 1.6;
  auto gauss_kernel = GaussKernel(radius, sigma);

  std::size_t height = 4;
  std::size_t width = 4;
  // clang-format off
  std::vector<Pixel> pixels = {
    1, 50, 100, 150,
    50, 100, 50, 125,
    100, 50, 100, 175,
    50, 100, 50, 4,
  };
  // clang-format on
  auto input_image = Image(height, width, pixels);
  auto output_image = Image(height, width, std::vector<Pixel>(height * width, 0));

  // clang-format off
  std::vector<Pixel> expected_pixels = {
    31, 56, 96, 124,
    56, 66, 98, 127,
    72, 73, 84, 94,
    71, 72, 68, 59,
  };
  // clang-format on

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&gauss_kernel));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_image));
  taskData->inputs_count.emplace_back(1);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&output_image));
  taskData->outputs_count.emplace_back(1);

  // Create Task
  ImageFilterGaussVerticalTask task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<Pixel> output_image_pixels = output_image.pixels();
  for (std::size_t i = 0; i < output_image_pixels.size(); i++) {
    ASSERT_EQ(output_image_pixels[i], expected_pixels[i]);
  }
}

TEST(kudinov_n_image_filter_gauss_vertical_seq, non_square_small_image) {
  // Create data
  std::size_t radius = 1;
  double sigma = 1.8;
  auto gauss_kernel = GaussKernel(radius, sigma);

  std::size_t height = 5;
  std::size_t width = 3;
  // clang-format off
  std::vector<Pixel> pixels = {
    10, 20, 30,
    40, 50, 60,
    70, 80, 90,
    100, 110, 120,
    130, 140, 150,
  };
  // clang-format on
  auto input_image = Image(height, width, pixels);
  auto output_image = Image(height, width, std::vector<Pixel>(height * width, 0));

  // clang-format off
  std::vector<Pixel> expected_pixels = {
    22, 29, 36,
    43, 49, 56,
    73, 80, 86,
    103, 110, 116,
    123, 130, 137,
  };
  // clang-format on

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&gauss_kernel));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_image));
  taskData->inputs_count.emplace_back(1);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&output_image));
  taskData->outputs_count.emplace_back(1);

  // Create Task
  ImageFilterGaussVerticalTask task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<Pixel> output_image_pixels = output_image.pixels();
  for (std::size_t i = 0; i < output_image_pixels.size(); i++) {
    ASSERT_EQ(output_image_pixels[i], expected_pixels[i]);
  }
}

TEST(kudinov_n_image_filter_gauss_vertical_seq, square_medium_image) {
  // Create data
  std::size_t radius = 2;
  double sigma = 1.8;
  auto gauss_kernel = GaussKernel(radius, sigma);

  std::size_t height = 10;
  std::size_t width = 10;
  // clang-format off
  std::vector<Pixel> pixels = {
    222, 216, 180,  92,  32, 152, 214, 164, 186,  13,
    219,  74, 134, 163,  34,  40, 248,  97, 218, 126,
     76,  11, 109,   6, 161, 206, 172, 203, 227, 164,
    240, 147, 171, 148, 208, 209, 218, 168,  16, 153,
     76, 154, 238, 138,  16, 179, 137, 168, 198, 227,
     57,  65, 113,  44, 144, 166,  34,  30,  16, 108,
    101,  45,  66,  45, 231,  27, 239, 148, 226,  67,
    251,   9,  23, 159, 124, 201, 110,  33, 138, 142,
    150,  40, 108,  21, 108,  94,  32, 182, 189, 232,
     23,  51,  45, 109, 159, 250,  48, 227,  61, 179,
  };
  // clang-format on
  auto input_image = Image(height, width, pixels);
  auto output_image = Image(height, width, std::vector<Pixel>(height * width, 0));

  // clang-format off
  std::vector<Pixel> expected_pixels = {
    184, 164, 134, 113, 118, 131, 157, 162, 140, 104,
    169, 150, 129, 117, 129, 141, 161, 164, 148, 123,
    151, 139, 127, 123, 137, 148, 163, 167, 160, 149,
    130, 125, 122, 127, 139, 145, 151, 153, 151, 153,
    114, 113, 117, 125, 136, 141, 146, 143, 141, 143,
    115, 110, 111, 118, 130, 131, 132, 128, 126, 129,
    107, 95, 94, 100, 114, 118, 124, 126, 131, 136,
    103, 85, 85, 95, 113, 122, 127, 127, 132, 141,
    92, 79, 82, 99, 116, 131, 136, 139, 146, 159,
    70, 68, 78, 105, 121, 139, 138, 140, 145, 167,
  };
  // clang-format on

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&gauss_kernel));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_image));
  taskData->inputs_count.emplace_back(1);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&output_image));
  taskData->outputs_count.emplace_back(1);

  // Create Task
  ImageFilterGaussVerticalTask task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<Pixel> output_image_pixels = output_image.pixels();
  for (std::size_t i = 0; i < output_image_pixels.size(); i++) {
    ASSERT_EQ(output_image_pixels[i], expected_pixels[i]);
  }
}

TEST(kudinov_n_image_filter_gauss_vertical_seq, non_square_medium_image) {
  // Create data
  std::size_t radius = 2;
  double sigma = 2.0;
  auto gauss_kernel = GaussKernel(radius, sigma);

  std::size_t height = 10;
  std::size_t width = 8;
  // clang-format off
  std::vector<Pixel> pixels = {
    100, 161, 132, 159, 181, 243,  22, 154,
    156, 208, 157,  29, 222,  11, 224,  13,
    28, 128,  71, 102, 132,   9, 223, 140,
    164, 228, 170, 226, 182,  56,  81,  61,
    62, 134, 222, 120,  48,  67,  59,  33,
    240,  46, 114,   0, 143, 115, 244, 137,
    19, 127,  37, 179, 170,  36,  12, 205,
    140, 251, 225, 192, 161, 100, 234, 123,
    130, 252, 163, 178, 178,  52, 130, 230,
    100, 100, 243,  47,   5, 112,  51,  83,
  };
  // clang-format on
  auto input_image = Image(height, width, pixels);
  auto output_image = Image(height, width, std::vector<Pixel>(height * width, 0));

  // clang-format off
  std::vector<Pixel> expected_pixels = {
    119, 126, 139, 148, 141, 138, 132, 120,
    126, 132, 143, 142, 135, 128, 120, 107,
    124, 133, 142, 134, 124, 112, 102, 90,
    132, 134, 136, 123, 117, 106, 100, 89,
    120, 124, 127, 119, 113, 106, 103, 102,
    131, 134, 137, 128, 121, 114, 112, 111,
    130, 135, 140, 132, 125, 124, 129, 135,
    136, 142, 146, 137, 127, 126, 133, 142,
    132, 144, 147, 138, 119, 110, 116, 128,
    139, 145, 139, 127, 108, 94, 101, 112,
  };
  // clang-format on

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&gauss_kernel));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_image));
  taskData->inputs_count.emplace_back(1);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&output_image));
  taskData->outputs_count.emplace_back(1);

  // Create Task
  ImageFilterGaussVerticalTask task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<Pixel> output_image_pixels = output_image.pixels();
  for (std::size_t i = 0; i < output_image_pixels.size(); i++) {
    ASSERT_EQ(output_image_pixels[i], expected_pixels[i]);
  }
}

TEST(kudinov_n_image_filter_gauss_vertical_seq, square_large_image) {
  // Create data
  std::size_t radius = 3;
  double sigma = 1.6;
  auto gauss_kernel = GaussKernel(radius, sigma);

  std::size_t height = 16;
  std::size_t width = 16;
  // clang-format off
  std::vector<Pixel> pixels = {
    154, 124, 124, 142, 206, 253,  91,  64, 205, 171, 164,  26, 200, 240, 109, 109,
    131, 190, 145, 117, 173,  15, 167, 156, 214,  24, 236, 209, 180, 87, 133,  50,
      1,  13, 166,  69, 115,  95, 225, 229,  22,  55,  79,   8,  79, 149, 100, 151,
    130, 183, 150,  72, 249, 135, 223, 144, 232, 115,  26, 193,  48, 32, 228,  71,
    149, 112, 105, 157, 180, 192,  53, 157, 128,  42, 188, 197,  87, 79,  42, 181,
    137,  42, 103, 178,  73, 149, 182,  71, 175, 225, 160, 189, 110, 178, 138,  46,
    173, 158, 140, 185, 168,  64, 162,  15, 110,  46, 152, 212,  57, 87,  60,  63,
    225, 234, 148,  33,  14, 241, 228, 135, 182,  61,  27,  60,  78, 169, 105,  21,
     95, 138,  62,  70, 157,  39,  22,  60,  50, 242,  90, 210, 241, 6, 246,  71,
     97, 210, 227, 211, 125,  91, 128, 207, 150, 165,  84, 192, 193, 180,  93, 216,
     29,   0,  24, 138,  28, 150, 166, 168, 204,  77, 220,  74, 218, 160, 108, 183,
     98,  46, 221, 231, 207, 170,  63, 102,  29,  17, 120, 122,  41, 74,  98, 171,
     79,  80, 145,  84, 211,  84, 147,  77, 128,  11, 225, 228,  38, 146, 148, 142,
    132,   0, 130, 231, 205, 224, 161, 148, 198,  82, 253,  33,  49, 189,  28, 161,
     78, 233,  55, 151,  24, 251,  94,  46,  76, 212,  43,  39, 206, 34,  89,  69,
    243, 137,  84, 206,  34,   4, 201, 108, 185,  85,  47, 142,  56, 42,   1, 217,
  };
  // clang-format on
  auto input_image = Image(height, width, pixels);
  auto output_image = Image(height, width, std::vector<Pixel>(height * width, 0));

  // clang-format off
  std::vector<Pixel> expected_pixels = {
    129, 132, 138, 147, 153, 154, 148, 144, 141, 139, 137, 139, 141, 141, 130, 117,
    118, 124, 131, 139, 146, 149, 147, 145, 138, 132, 128, 128, 130, 129, 122, 113,
    110, 118, 126, 134, 142, 148, 149, 145, 136, 127, 121, 119, 119, 119, 116, 112,
    114, 119, 126, 134, 142, 147, 147, 143, 134, 126, 121, 118, 114, 112, 111, 110,
    126, 127, 129, 135, 140, 144, 143, 138, 133, 128, 126, 122, 115, 109, 105, 102,
    139, 134, 132, 133, 135, 137, 134, 130, 128, 128, 128, 126, 118, 109, 100, 92,
    150, 143, 135, 131, 128, 128, 127, 124, 124, 125, 128, 128, 122, 112, 100, 87,
    146, 140, 132, 125, 121, 121, 122, 122, 123, 125, 127, 130, 127, 120, 108, 96,
    131, 132, 129, 124, 120, 119, 121, 123, 125, 127, 130, 135, 135, 130, 123, 115,
    111, 118, 124, 126, 124, 122, 123, 124, 126, 128, 133, 137, 139, 138, 136, 137,
    95, 107, 121, 131, 133, 131, 127, 125, 124, 126, 130, 134, 136, 137, 140, 147,
    90, 104, 122, 137, 142, 139, 129, 122, 119, 120, 125, 127, 126, 128, 134, 144,
    99, 109, 126, 141, 147, 144, 133, 123, 118, 117, 119, 118, 115, 116, 123, 135,
    115, 118, 127, 137, 142, 140, 132, 124, 120, 117, 114, 109, 104, 104, 111, 125,
    140, 134, 131, 131, 129, 129, 128, 124, 121, 115, 108, 99, 93, 93, 104, 122,
    165, 148, 132, 121, 114, 115, 121, 123, 122, 113, 102, 91, 83, 85, 101, 127,
  };
  // clang-format on

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&gauss_kernel));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&input_image));
  taskData->inputs_count.emplace_back(1);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&output_image));
  taskData->outputs_count.emplace_back(1);

  // Create Task
  ImageFilterGaussVerticalTask task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<Pixel> output_image_pixels = output_image.pixels();
  for (std::size_t i = 0; i < output_image_pixels.size(); i++) {
    ASSERT_EQ(output_image_pixels[i], expected_pixels[i]);
  }
}
