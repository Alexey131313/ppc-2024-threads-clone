// Copyright 2024 Savotina Valeria
#include <gtest/gtest.h>

#include "seq/savotina_v_grahams_alg/include/ops_seq.hpp"

// Points are arranged in a chaotic order (type of x and y is int)
TEST(savotina_v_grahams_alg_seq, Test1) {
  // Create data
  std::vector<SavotinaPoint> points = {SavotinaPoint(-8, 4),   SavotinaPoint(-4, 6),   SavotinaPoint(-12, 2),
                                       SavotinaPoint(-6, -2),  SavotinaPoint(-10, -4), SavotinaPoint(-4, 2),
                                       SavotinaPoint(-6, 6),   SavotinaPoint(-8, 8),   SavotinaPoint(-10, 6),
                                       SavotinaPoint(-8, 2),   SavotinaPoint(-6, 2),   SavotinaPoint(-10, 0),
                                       SavotinaPoint(-14, -2), SavotinaPoint(-16, -4), SavotinaPoint(-14, -4)};
  std::vector<SavotinaPoint> res = {SavotinaPoint(-16, -4), SavotinaPoint(-10, -4), SavotinaPoint(-6, -2),
                                    SavotinaPoint(-4, 2),   SavotinaPoint(-4, 6),   SavotinaPoint(-8, 8),
                                    SavotinaPoint(-10, 6)};
  std::vector<SavotinaPoint> mch(res.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> dataGrahamsAlgSeq = std::make_shared<ppc::core::TaskData>();
  dataGrahamsAlgSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  dataGrahamsAlgSeq->inputs_count.emplace_back(points.size());
  dataGrahamsAlgSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(mch.data()));
  dataGrahamsAlgSeq->outputs_count.emplace_back(mch.size());

  // Create Task
  SavotinaGrahamsAlgorithmSequential testGrahamsAlgSeq(dataGrahamsAlgSeq);
  ASSERT_EQ(testGrahamsAlgSeq.validation(), true);
  testGrahamsAlgSeq.pre_processing();
  testGrahamsAlgSeq.run();
  testGrahamsAlgSeq.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(mch[i], res[i]);
  }
}

// Points are arranged in a chaotic order (type of x and y is double)
TEST(savotina_v_grahams_alg_seq, Test2) {
  // Create data
  std::vector<SavotinaPoint> points = {
      SavotinaPoint(-0.5, 2.2),  SavotinaPoint(1.6, 1.3),  SavotinaPoint(0.3, -0.6),  SavotinaPoint(0.1, 1.3),
      SavotinaPoint(-1.8, 1.42), SavotinaPoint(-0.3, 0.6), SavotinaPoint(-0.5, -1.2), SavotinaPoint(1.2, -0.8),
      SavotinaPoint(0.7, 0.4),   SavotinaPoint(1.1, 1.9),  SavotinaPoint(0.4, -1.2),  SavotinaPoint(-1.9, 0.4),
      SavotinaPoint(-0.6, -0.3), SavotinaPoint(1.8, 0.5),  SavotinaPoint(-1.4, -0.7), SavotinaPoint(-0.9, 1.1),
      SavotinaPoint(-1.2, 1.9),  SavotinaPoint(0.4, 2.2),  SavotinaPoint(1.7, -0.1)};
  std::vector<SavotinaPoint> res = {SavotinaPoint(-1.9, 0.4), SavotinaPoint(-1.4, -0.7), SavotinaPoint(-0.5, -1.2),
                                    SavotinaPoint(0.4, -1.2), SavotinaPoint(1.2, -0.8),  SavotinaPoint(1.7, -0.1),
                                    SavotinaPoint(1.8, 0.5),  SavotinaPoint(1.6, 1.3),   SavotinaPoint(1.1, 1.9),
                                    SavotinaPoint(0.4, 2.2),  SavotinaPoint(-0.5, 2.2),  SavotinaPoint(-1.2, 1.9),
                                    SavotinaPoint(-1.8, 1.42)};
  std::vector<SavotinaPoint> mch(res.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> dataGrahamsAlgSeq = std::make_shared<ppc::core::TaskData>();
  dataGrahamsAlgSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  dataGrahamsAlgSeq->inputs_count.emplace_back(points.size());
  dataGrahamsAlgSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(mch.data()));
  dataGrahamsAlgSeq->outputs_count.emplace_back(mch.size());

  // Create Task
  SavotinaGrahamsAlgorithmSequential testGrahamsAlgSeq(dataGrahamsAlgSeq);
  ASSERT_EQ(testGrahamsAlgSeq.validation(), true);
  testGrahamsAlgSeq.pre_processing();
  testGrahamsAlgSeq.run();
  testGrahamsAlgSeq.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(mch[i], res[i]);
  }
}

// Points are located on the same straight line
TEST(savotina_v_grahams_alg_seq, Test3) {
  // Create data
  std::vector<SavotinaPoint> points = {SavotinaPoint(-1.4, 2.2), SavotinaPoint(2.2, 2.2),  SavotinaPoint(-0.6, 2.2),
                                       SavotinaPoint(-2.3, 2.2), SavotinaPoint(2.7, 2.2),  SavotinaPoint(0.6, 2.2),
                                       SavotinaPoint(1.7, 2.2),  SavotinaPoint(-3.4, 2.2), SavotinaPoint(5.3, 2.2),
                                       SavotinaPoint(4.4, 2.2),  SavotinaPoint(3.4, 2.2)};
  std::vector<SavotinaPoint> res = {SavotinaPoint(-3.4, 2.2), SavotinaPoint(5.3, 2.2)};
  std::vector<SavotinaPoint> mch(res.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> dataGrahamsAlgSeq = std::make_shared<ppc::core::TaskData>();
  dataGrahamsAlgSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  dataGrahamsAlgSeq->inputs_count.emplace_back(points.size());
  dataGrahamsAlgSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(mch.data()));
  dataGrahamsAlgSeq->outputs_count.emplace_back(mch.size());

  // Create Task
  SavotinaGrahamsAlgorithmSequential testGrahamsAlgSeq(dataGrahamsAlgSeq);
  ASSERT_EQ(testGrahamsAlgSeq.validation(), true);
  testGrahamsAlgSeq.pre_processing();
  testGrahamsAlgSeq.run();
  testGrahamsAlgSeq.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(mch[i], res[i]);
  }
}

// Only one point
TEST(savotina_v_grahams_alg_seq, Test4) {
  // Create data
  std::vector<SavotinaPoint> points = {SavotinaPoint(3.5, 4.7)};
  std::vector<SavotinaPoint> res = {SavotinaPoint(3.5, 4.7)};
  std::vector<SavotinaPoint> mch(res.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> dataGrahamsAlgSeq = std::make_shared<ppc::core::TaskData>();
  dataGrahamsAlgSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  dataGrahamsAlgSeq->inputs_count.emplace_back(points.size());
  dataGrahamsAlgSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(mch.data()));
  dataGrahamsAlgSeq->outputs_count.emplace_back(mch.size());

  // Create Task
  SavotinaGrahamsAlgorithmSequential testGrahamsAlgSeq(dataGrahamsAlgSeq);
  ASSERT_EQ(testGrahamsAlgSeq.validation(), true);
  testGrahamsAlgSeq.pre_processing();
  testGrahamsAlgSeq.run();
  testGrahamsAlgSeq.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(mch[i], res[i]);
  }
}

// Only two points
TEST(savotina_v_grahams_alg_seq, Test5) {
  // Create data
  std::vector<SavotinaPoint> points = {SavotinaPoint(2.4, -2.9), SavotinaPoint(-1.8, 4.2)};
  std::vector<SavotinaPoint> res = {SavotinaPoint(-1.8, 4.2), SavotinaPoint(2.4, -2.9)};
  std::vector<SavotinaPoint> mch(res.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> dataGrahamsAlgSeq = std::make_shared<ppc::core::TaskData>();
  dataGrahamsAlgSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  dataGrahamsAlgSeq->inputs_count.emplace_back(points.size());
  dataGrahamsAlgSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(mch.data()));
  dataGrahamsAlgSeq->outputs_count.emplace_back(mch.size());

  // Create Task
  SavotinaGrahamsAlgorithmSequential testGrahamsAlgSeq(dataGrahamsAlgSeq);
  ASSERT_EQ(testGrahamsAlgSeq.validation(), true);
  testGrahamsAlgSeq.pre_processing();
  testGrahamsAlgSeq.run();
  testGrahamsAlgSeq.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(mch[i], res[i]);
  }
}

// No points
TEST(savotina_v_grahams_alg_seq, Test6) {
  // Create data
  std::vector<SavotinaPoint> points = {};
  std::vector<SavotinaPoint> res = {};
  std::vector<SavotinaPoint> mch(res.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> dataGrahamsAlgSeq = std::make_shared<ppc::core::TaskData>();
  dataGrahamsAlgSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  dataGrahamsAlgSeq->inputs_count.emplace_back(points.size());
  dataGrahamsAlgSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(mch.data()));
  dataGrahamsAlgSeq->outputs_count.emplace_back(mch.size());

  // Create Task
  SavotinaGrahamsAlgorithmSequential testGrahamsAlgSeq(dataGrahamsAlgSeq);
  ASSERT_EQ(testGrahamsAlgSeq.validation(), true);
  testGrahamsAlgSeq.pre_processing();
  testGrahamsAlgSeq.run();
  testGrahamsAlgSeq.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(mch[i], res[i]);
  }
}

// A lot of same points (in example: 10 points)
TEST(savotina_v_grahams_alg_seq, Test7) {
  // Create data
  std::vector<SavotinaPoint> points = {SavotinaPoint(4.4, 1.7), SavotinaPoint(4.4, 1.7), SavotinaPoint(4.4, 1.7),
                                       SavotinaPoint(4.4, 1.7), SavotinaPoint(4.4, 1.7), SavotinaPoint(4.4, 1.7),
                                       SavotinaPoint(4.4, 1.7), SavotinaPoint(4.4, 1.7), SavotinaPoint(4.4, 1.7),
                                       SavotinaPoint(4.4, 1.7)};
  std::vector<SavotinaPoint> res = {SavotinaPoint(4.4, 1.7)};
  std::vector<SavotinaPoint> mch(res.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> dataGrahamsAlgSeq = std::make_shared<ppc::core::TaskData>();
  dataGrahamsAlgSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  dataGrahamsAlgSeq->inputs_count.emplace_back(points.size());
  dataGrahamsAlgSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(mch.data()));
  dataGrahamsAlgSeq->outputs_count.emplace_back(mch.size());

  // Create Task
  SavotinaGrahamsAlgorithmSequential testGrahamsAlgSeq(dataGrahamsAlgSeq);
  ASSERT_EQ(testGrahamsAlgSeq.validation(), true);
  testGrahamsAlgSeq.pre_processing();
  testGrahamsAlgSeq.run();
  testGrahamsAlgSeq.post_processing();

  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(mch[i], res[i]);
  }
}
