// Copyright 2024 Kriseev Mikhail
#include <gtest/gtest.h>

#include "seq/kriseev_m_convex_hull_graham/include/kriseev_m_convex_hull_seq.hpp"

TEST(kriseev_m_convex_hull_graham_seq, ThreePoints) {
  std::vector<double> pX;
  pX.emplace_back(-2.0);
  pX.emplace_back(4.0);
  pX.emplace_back(1.0);
  std::vector<double> pY;
  pY.emplace_back(-2.0);
  pY.emplace_back(0.0);
  pY.emplace_back(3.0);

  std::vector<double> outX(pX.size());
  std::vector<double> outY(pY.size());

  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pX.data()));
  data->inputs_count.emplace_back(pX.size());
  data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pY.data()));
  data->inputs_count.emplace_back(pY.size());

  data->outputs.emplace_back(reinterpret_cast<uint8_t *>(outX.data()));
  data->outputs_count.emplace_back(outX.size());
  data->outputs.emplace_back(reinterpret_cast<uint8_t *>(outY.data()));
  data->outputs_count.emplace_back(outY.size());

  KriseevMTaskSeq::ConvexHullTask task(data);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  EXPECT_EQ(pX.size(), data->outputs_count[0]);
  EXPECT_EQ(pY.size(), data->outputs_count[1]);

  for (size_t i = 0; i < data->outputs_count[0]; ++i) {
    EXPECT_EQ(pX[i], outX[i]);
    EXPECT_EQ(pY[i], outY[i]);
  }
}

TEST(kriseev_m_convex_hull_graham_seq, Square) {
  std::vector<double> pX;
  pX.emplace_back(2.0);
  pX.emplace_back(5.0);
  pX.emplace_back(5.0);
  pX.emplace_back(2.0);
  std::vector<double> pY;
  pY.emplace_back(5.0);
  pY.emplace_back(2.0);
  pY.emplace_back(5.0);
  pY.emplace_back(2.0);

  std::vector<double> outX(pX.size());
  std::vector<double> outY(pY.size());

  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pX.data()));
  data->inputs_count.emplace_back(pX.size());
  data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pY.data()));
  data->inputs_count.emplace_back(pY.size());

  data->outputs.emplace_back(reinterpret_cast<uint8_t *>(outX.data()));
  data->outputs_count.emplace_back(outX.size());
  data->outputs.emplace_back(reinterpret_cast<uint8_t *>(outY.data()));
  data->outputs_count.emplace_back(outY.size());

  std::vector<KriseevMTaskSeq::Point> expectedPoints = {
      {pX[1], pY[1]},
      {pX[2], pY[2]},
      {pX[0], pY[0]},
      {pX[3], pY[3]},
  };

  KriseevMTaskSeq::ConvexHullTask task(data);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  EXPECT_EQ(expectedPoints.size(), data->outputs_count[0]);
  EXPECT_EQ(expectedPoints.size(), data->outputs_count[1]);

  for (size_t i = 0; i < data->outputs_count[0]; ++i) {
    EXPECT_EQ(expectedPoints[i].first, outX[i]);
    EXPECT_EQ(expectedPoints[i].second, outY[i]);
  }
}

TEST(kriseev_m_convex_hull_graham_seq, SquareWithPointInTheMiddle) {
  std::vector<double> pX;
  pX.emplace_back(2.0);
  pX.emplace_back(5.0);
  pX.emplace_back(5.0);
  pX.emplace_back(3.5);
  pX.emplace_back(2.0);
  std::vector<double> pY;
  pY.emplace_back(5.0);
  pY.emplace_back(2.0);
  pY.emplace_back(5.0);
  pY.emplace_back(4.0);
  pY.emplace_back(2.0);

  std::vector<double> outX(pX.size());
  std::vector<double> outY(pY.size());

  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pX.data()));
  data->inputs_count.emplace_back(pX.size());
  data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pY.data()));
  data->inputs_count.emplace_back(pY.size());

  data->outputs.emplace_back(reinterpret_cast<uint8_t *>(outX.data()));
  data->outputs_count.emplace_back(outX.size());
  data->outputs.emplace_back(reinterpret_cast<uint8_t *>(outY.data()));
  data->outputs_count.emplace_back(outY.size());

  std::vector<KriseevMTaskSeq::Point> expectedPoints = {
      {pX[1], pY[1]},
      {pX[2], pY[2]},
      {pX[0], pY[0]},
      {pX[4], pY[4]},
  };

  KriseevMTaskSeq::ConvexHullTask task(data);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  EXPECT_EQ(expectedPoints.size(), data->outputs_count[0]);
  EXPECT_EQ(expectedPoints.size(), data->outputs_count[1]);

  for (size_t i = 0; i < data->outputs_count[0]; ++i) {
    EXPECT_EQ(expectedPoints[i].first, outX[i]);
    EXPECT_EQ(expectedPoints[i].second, outY[i]);
  }
}

TEST(kriseev_m_convex_hull_graham_seq, SquareWithPointOnEdge) {
  std::vector<double> pX;
  pX.emplace_back(2.0);
  pX.emplace_back(5.0);
  pX.emplace_back(5.0);
  pX.emplace_back(2.0);
  pX.emplace_back(2.0);
  std::vector<double> pY;
  pY.emplace_back(5.0);
  pY.emplace_back(2.0);
  pY.emplace_back(5.0);
  pY.emplace_back(4.0);
  pY.emplace_back(2.0);

  std::vector<double> outX(pX.size());
  std::vector<double> outY(pY.size());

  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pX.data()));
  data->inputs_count.emplace_back(pX.size());
  data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pY.data()));
  data->inputs_count.emplace_back(pY.size());

  data->outputs.emplace_back(reinterpret_cast<uint8_t *>(outX.data()));
  data->outputs_count.emplace_back(outX.size());
  data->outputs.emplace_back(reinterpret_cast<uint8_t *>(outY.data()));
  data->outputs_count.emplace_back(outY.size());

  std::vector<KriseevMTaskSeq::Point> expectedPoints = {
      {pX[1], pY[1]},
      {pX[2], pY[2]},
      {pX[0], pY[0]},
      {pX[4], pY[4]},
  };

  KriseevMTaskSeq::ConvexHullTask task(data);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  EXPECT_EQ(expectedPoints.size(), data->outputs_count[0]);
  EXPECT_EQ(expectedPoints.size(), data->outputs_count[1]);

  for (size_t i = 0; i < data->outputs_count[0]; ++i) {
    EXPECT_EQ(expectedPoints[i].first, outX[i]);
    EXPECT_EQ(expectedPoints[i].second, outY[i]);
  }
}

TEST(kriseev_m_convex_hull_graham_seq, PentagonWithPointsInside) {
  std::vector<double> pX;
  pX.emplace_back(1.5);

  pX.emplace_back(2.0);

  pX.emplace_back(2.2);
  pX.emplace_back(4.1);
  pX.emplace_back(2.0);

  pX.emplace_back(5.0);

  pX.emplace_back(3.7);
  pX.emplace_back(4.3);

  pX.emplace_back(5.0);

  pX.emplace_back(5.0);
  pX.emplace_back(3.5);
  pX.emplace_back(2.7);
  pX.emplace_back(3.0);

  pX.emplace_back(1.0);
  pX.emplace_back(2.0);
  std::vector<double> pY;
  pY.emplace_back(3.5);

  pY.emplace_back(5.0);

  pY.emplace_back(3.2);
  pY.emplace_back(4.2);
  pY.emplace_back(2.4);

  pY.emplace_back(2.0);

  pY.emplace_back(3.2);
  pY.emplace_back(5.0);

  pY.emplace_back(5.0);

  pY.emplace_back(2.9);
  pY.emplace_back(4.8);
  pY.emplace_back(2.5);
  pY.emplace_back(4.6);

  pY.emplace_back(4.0);
  pY.emplace_back(2.0);

  std::vector<double> outX(pX.size());
  std::vector<double> outY(pY.size());

  std::shared_ptr<ppc::core::TaskData> data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pX.data()));
  data->inputs_count.emplace_back(pX.size());
  data->inputs.emplace_back(reinterpret_cast<uint8_t *>(pY.data()));
  data->inputs_count.emplace_back(pY.size());

  data->outputs.emplace_back(reinterpret_cast<uint8_t *>(outX.data()));
  data->outputs_count.emplace_back(outX.size());
  data->outputs.emplace_back(reinterpret_cast<uint8_t *>(outY.data()));
  data->outputs_count.emplace_back(outY.size());

  std::vector<KriseevMTaskSeq::Point> expectedPoints = {
      {pX[5], pY[5]}, {pX[8], pY[8]}, {pX[1], pY[1]}, {pX[13], pY[13]}, {pX[14], pY[14]},
  };

  KriseevMTaskSeq::ConvexHullTask task(data);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  EXPECT_EQ(expectedPoints.size(), data->outputs_count[0]);
  EXPECT_EQ(expectedPoints.size(), data->outputs_count[1]);

  for (size_t i = 0; i < data->outputs_count[0]; ++i) {
    EXPECT_EQ(expectedPoints[i].first, outX[i]);
    EXPECT_EQ(expectedPoints[i].second, outY[i]);
  }
}
