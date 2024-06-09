// Copyright 2024 Larin Konstantin
#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#include "seq/larin_k_integral/include/integral.hpp"
using namespace larin;

TEST(larin_k_integral, int_line_area) {
  // Create data
  auto x_min = [](auto&&) { return 0; };
  auto x_max = [](auto&&) { return 4; };
  auto func = [](const std::vector<num_t>& x) { return x[0]; };
  num_t expected_value = 8;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  useless_class_that_exists_because_of_disgusting_api test({{x_min, x_max}}, func, taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  EXPECT_NEAR(test.get_result(), expected_value, 1e-6);
}

TEST(larin_k_integral, int_sin_area) {
  // Create data
  auto c_0 = [](auto&&) { return 0; };
  auto c_2pi = [](auto&&) { return M_PI * 2; };
  auto sin_func = [](const std::vector<num_t>& x) { return std::sin(x[0]); };
  num_t expected_value = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  useless_class_that_exists_because_of_disgusting_api test({{c_0, c_2pi}}, sin_func, taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  EXPECT_NEAR(test.get_result(), expected_value, 1e-6);
}

TEST(larin_k_integral, int_square_area) {
  // Create data
  auto c_1 = [](auto&&) { return 1; };
  auto c_n1 = [](auto&&) { return -1; };
  num_t expected_value = 4;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  useless_class_that_exists_because_of_disgusting_api test({{c_n1, c_1}, {c_n1, c_1}}, c_1, taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  EXPECT_NEAR(test.get_result(), expected_value, 1e-3);
}

TEST(larin_k_integral, int_circle_area) {
  // Create data
  auto c_1 = [](auto&&) { return 1; };
  auto c_n1 = [](auto&&) { return -1; };

  auto y_min = [](const std::vector<num_t>& x) { return -sqrt(1 - x[0] * x[0]); };
  auto y_max = [](const std::vector<num_t>& x) { return sqrt(1 - x[0] * x[0]); };
  num_t expected_value = M_PI;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  useless_class_that_exists_because_of_disgusting_api test({{c_n1, c_1}, {y_min, y_max}}, c_1, taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  EXPECT_NEAR(test.get_result(), expected_value, 1e-3);
}

TEST(larin_k_integral, int_ball_area) {
  // Create data
  limit_t phi_limits{func_t{[](auto&&) { return 0; }}, func_t{[](auto&&) { return M_PI; }}};
  limit_t theta_limits{func_t{[](auto&&) { return 0; }}, func_t{[](auto&&) { return M_PI; }}};
  limit_t r_limits{func_t{[](auto&&) { return 0; }}, func_t{[](auto&&) { return 1; }}};

  func_t func = [](const std::vector<num_t>& x) {
    auto r = x[2];
    auto theta = x[1];
    return r * r * std::sin(theta);
  };

  num_t expected_value = 2 / 3. * M_PI;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  useless_class_that_exists_because_of_disgusting_api test({phi_limits, theta_limits, r_limits}, func, taskDataSeq);
  test.set_step(1e-2);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  EXPECT_NEAR(test.get_result(), expected_value, 1e-2);
}
