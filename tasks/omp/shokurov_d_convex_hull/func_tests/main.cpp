// Copyright 2024 Shokurov Daniil
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include "omp/shokurov_d_convex_hull/include/ops_omp.hpp"

TEST(shokurov_d_convex_hull_omp, Test_one_point) {
  // Create data
  std::vector<pair<double, double>> in;
  in.emplace_back(0, 0);
  std::vector<pair<double, double>> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create answer
  std::vector<pair<double, double>> ans;
  ans.emplace_back(0.0, 0.0);

  // Create Task
  ConvexHullSequential test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  ASSERT_EQ(test.pre_processing(), true);
  ASSERT_EQ(test.run(), true);
  ASSERT_EQ(test.post_processing(), true);

  size_t k = taskDataSeq->outputs_count[0];
  ASSERT_EQ(ans.size(), k);

  auto *_out = reinterpret_cast<pair<double, double> *>(taskDataSeq->outputs[0]);

  sort(_out, _out + k);
  sort(ans.begin(), ans.end());

  for (size_t i = 0; i < k; ++i) {
    ASSERT_DOUBLE_EQ(_out[i].first, ans[i].first);
    ASSERT_DOUBLE_EQ(_out[i].second, ans[i].second);
  }
}

TEST(shokurov_d_convex_hull_omp, Test_many_equals_point) {
  // Create data
  std::vector<pair<double, double>> in;
  in.emplace_back(0, 0);
  in.emplace_back(0, 0);
  in.emplace_back(0, 0);
  in.emplace_back(0, 0);
  in.emplace_back(0, 0);
  std::vector<pair<double, double>> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create answer
  std::vector<pair<double, double>> ans;
  ans.emplace_back(0, 0);

  // Create Task
  ConvexHullSequential test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  ASSERT_EQ(test.pre_processing(), true);
  ASSERT_EQ(test.run(), true);
  ASSERT_EQ(test.post_processing(), true);

  size_t k = taskDataSeq->outputs_count[0];
  ASSERT_EQ(ans.size(), k);

  auto *_out = reinterpret_cast<pair<double, double> *>(taskDataSeq->outputs[0]);

  std::sort(_out, _out + k);
  std::sort(ans.begin(), ans.end());

  for (size_t i = 0; i < k; ++i) {
    ASSERT_DOUBLE_EQ(_out[i].first, ans[i].first);
    ASSERT_DOUBLE_EQ(_out[i].second, ans[i].second);
  }
}

TEST(shokurov_d_convex_hull_omp, Test_two_point) {
  // Create data
  std::vector<pair<double, double>> in;
  in.emplace_back(1, 1);
  in.emplace_back(0, 0);
  std::vector<pair<double, double>> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create answer
  std::vector<pair<double, double>> ans;
  ans.emplace_back(0, 0);
  ans.emplace_back(1, 1);
  // Create Task
  ConvexHullSequential test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  ASSERT_EQ(test.pre_processing(), true);
  ASSERT_EQ(test.run(), true);
  ASSERT_EQ(test.post_processing(), true);

  size_t k = taskDataSeq->outputs_count[0];
  ASSERT_EQ(ans.size(), k);

  auto *_out = reinterpret_cast<pair<double, double> *>(taskDataSeq->outputs[0]);

  std::sort(_out, _out + k);
  std::sort(ans.begin(), ans.end());

  for (size_t i = 0; i < k; ++i) {
    ASSERT_DOUBLE_EQ(_out[i].first, ans[i].first);
    ASSERT_DOUBLE_EQ(_out[i].second, ans[i].second);
  }
}

TEST(shokurov_d_convex_hull_omp, Test_line_segment) {
  // Create data
  std::vector<pair<double, double>> in;
  in.emplace_back(1, 1);
  in.emplace_back(0.75, 0.75);
  in.emplace_back(0.5, 0.5);
  in.emplace_back(0.25, 0.25);
  in.emplace_back(0.8, 0.8);
  in.emplace_back(0.15, 0.15);
  in.emplace_back(0.12, 0.12);
  in.emplace_back(0.99, 0.99);
  in.emplace_back(0, 0);
  std::vector<pair<double, double>> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create answer
  std::vector<pair<double, double>> ans;
  ans.emplace_back(0, 0);
  ans.emplace_back(1, 1);
  // Create Task
  ConvexHullSequential test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  ASSERT_EQ(test.pre_processing(), true);
  ASSERT_EQ(test.run(), true);
  ASSERT_EQ(test.post_processing(), true);

  size_t k = taskDataSeq->outputs_count[0];
  ASSERT_EQ(ans.size(), k);

  auto *_out = reinterpret_cast<pair<double, double> *>(taskDataSeq->outputs[0]);

  std::sort(_out, _out + k);
  std::sort(ans.begin(), ans.end());

  for (size_t i = 0; i < k; ++i) {
    ASSERT_DOUBLE_EQ(_out[i].first, ans[i].first);
    ASSERT_DOUBLE_EQ(_out[i].second, ans[i].second);
  }
}

TEST(shokurov_d_convex_hull_omp, Test_line_segment_2) {
  // Create data
  std::vector<pair<double, double>> in;
  in.emplace_back(1, 0);
  in.emplace_back(0.75, 0);
  in.emplace_back(0.5, 0);
  in.emplace_back(0.25, 0);
  in.emplace_back(0.8, 0);
  in.emplace_back(0.15, 0);
  in.emplace_back(0.12, 0);
  in.emplace_back(0.99, 0);
  in.emplace_back(0, 0);
  std::vector<pair<double, double>> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create answer
  std::vector<pair<double, double>> ans;
  ans.emplace_back(0, 0);
  ans.emplace_back(1, 0);
  // Create Task
  ConvexHullSequential test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  ASSERT_EQ(test.pre_processing(), true);
  ASSERT_EQ(test.run(), true);
  ASSERT_EQ(test.post_processing(), true);

  size_t k = taskDataSeq->outputs_count[0];
  ASSERT_EQ(ans.size(), k);

  auto *_out = reinterpret_cast<pair<double, double> *>(taskDataSeq->outputs[0]);

  std::sort(_out, _out + k);
  std::sort(ans.begin(), ans.end());

  for (size_t i = 0; i < k; ++i) {
    ASSERT_DOUBLE_EQ(_out[i].first, ans[i].first);
    ASSERT_DOUBLE_EQ(_out[i].second, ans[i].second);
  }
}

TEST(shokurov_d_convex_hull_omp, Test_square) {
  // Create data
  std::vector<pair<double, double>> in;
  in.emplace_back(0, 0);
  in.emplace_back(0, 1);
  in.emplace_back(1, 0);
  in.emplace_back(1, 1);
  in.emplace_back(0.5, 0.5);
  in.emplace_back(0.1, 0.6);
  in.emplace_back(0.2, 0.2);
  in.emplace_back(0.9, 0.9);
  in.emplace_back(0.1, 0.1);
  std::vector<pair<double, double>> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create answer
  std::vector<pair<double, double>> ans;
  ans.emplace_back(0, 0);
  ans.emplace_back(0, 1);
  ans.emplace_back(1, 0);
  ans.emplace_back(1, 1);

  // Create Task
  ConvexHullSequential test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  ASSERT_EQ(test.pre_processing(), true);
  ASSERT_EQ(test.run(), true);
  ASSERT_EQ(test.post_processing(), true);

  size_t k = taskDataSeq->outputs_count[0];
  ASSERT_EQ(ans.size(), k);

  auto *_out = reinterpret_cast<pair<double, double> *>(taskDataSeq->outputs[0]);

  std::sort(_out, _out + k);
  std::sort(ans.begin(), ans.end());

  for (size_t i = 0; i < k; ++i) {
    ASSERT_DOUBLE_EQ(_out[i].first, ans[i].first);
    ASSERT_DOUBLE_EQ(_out[i].second, ans[i].second);
  }
}

TEST(shokurov_d_convex_hull_omp, Test_triangle) {
  // Create data
  std::vector<pair<double, double>> in;
  in.emplace_back(0, 0);
  in.emplace_back(1, 0);
  in.emplace_back(0, 1);
  in.emplace_back(0.1, 0.1);
  in.emplace_back(0.2, 0.2);
  in.emplace_back(0.5, 0.5);
  in.emplace_back(0.1, 0.05);

  std::vector<pair<double, double>> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create answer
  std::vector<pair<double, double>> ans;
  ans.emplace_back(0, 0);
  ans.emplace_back(1, 0);
  ans.emplace_back(0, 1);

  // Create Task
  ConvexHullSequential test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  ASSERT_EQ(test.pre_processing(), true);
  ASSERT_EQ(test.run(), true);
  ASSERT_EQ(test.post_processing(), true);

  size_t k = taskDataSeq->outputs_count[0];
  ASSERT_EQ(ans.size(), k);

  auto *_out = reinterpret_cast<pair<double, double> *>(taskDataSeq->outputs[0]);

  std::sort(_out, _out + k);
  std::sort(ans.begin(), ans.end());

  for (size_t i = 0; i < k; ++i) {
    ASSERT_DOUBLE_EQ(_out[i].first, ans[i].first);
    ASSERT_DOUBLE_EQ(_out[i].second, ans[i].second);
  }
}

TEST(shokurov_d_convex_hull_omp, Test_random) {
  // Create data
  std::vector<pair<double, double>> in;
  std::vector<pair<double, double>> ans;
  ans.emplace_back(0, 0);
  ans.emplace_back(1, 0);
  ans.emplace_back(5, 5);
  ans.emplace_back(0, 8);
  ans.emplace_back(-3, 1);
  size_t co = 5;
  size_t n = 100;
  for (size_t i = 0; i < co; ++i) {
    in.emplace_back(ans[i]);
  }

  std::mt19937 gen(1.0);
  std::uniform_real_distribution<> dis(0.1, 0.9);

  for (size_t i = 0; i < n; ++i) {
    size_t x = i % co;
    size_t y = (i + 1) % co;
    double a = dis(gen);
    pair<double, double> np;
    np.first = ans[x].first * a + (1.0 - a) * ans[y].first;
    np.second = ans[x].second * a + (1.0 - a) * ans[y].second;
    in.emplace_back(np);
  }
  std::vector<pair<double, double>> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ConvexHullSequential test(taskDataSeq);
  ASSERT_EQ(test.validation(), true);
  ASSERT_EQ(test.pre_processing(), true);
  ASSERT_EQ(test.run(), true);
  ASSERT_EQ(test.post_processing(), true);

  // Check answer
  size_t k = taskDataSeq->outputs_count[0];
  ASSERT_EQ(ans.size(), k);

  auto *_out = reinterpret_cast<pair<double, double> *>(taskDataSeq->outputs[0]);

  std::sort(_out, _out + k);
  std::sort(ans.begin(), ans.end());

  for (size_t i = 0; i < k; ++i) {
    ASSERT_DOUBLE_EQ(_out[i].first, ans[i].first);
    ASSERT_DOUBLE_EQ(_out[i].second, ans[i].second);
  }
}
