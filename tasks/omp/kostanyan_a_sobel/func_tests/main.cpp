// Copyright 2023 Kostanyan Arsen
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "omp/kostanyan_a_sobel/include/ops_omp.hpp"

TEST(kostanyan_a_sobel_omp, Test_EdgeDetection) {
  const int n = 10;
  const int m = 10;
  const uint8_t min = 0;
  const uint8_t max = 255;

  // Create data
  std::vector<int> in(2);
  in[0] = n;
  in[1] = m;
  std::vector<uint8_t> pict = kostanyan_omp_sobel::kostanyan_getRandomPicture(n, m, min, max);
  std::vector<uint8_t> out_seq(n * m, 0);
  std::vector<uint8_t> out_omp(n * m, 0);

  // Create TaskData for Sequential
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(pict.data());
  taskDataSeq->inputs_count.emplace_back(pict.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  // Create TaskData for OpenMP
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataOmp->inputs_count.emplace_back(in.size());
  taskDataOmp->inputs.emplace_back(pict.data());
  taskDataOmp->inputs_count.emplace_back(pict.size());
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataOmp->outputs_count.emplace_back(out_omp.size());

  // Create Task for Sequential
  kostanyan_omp_sobel::Kostanyan_EdgeDetectionSequential kostanyan_EdgeDetectionSequential(taskDataSeq);
  ASSERT_EQ(kostanyan_EdgeDetectionSequential.validation(), true);
  kostanyan_EdgeDetectionSequential.pre_processing();
  kostanyan_EdgeDetectionSequential.run();
  kostanyan_EdgeDetectionSequential.post_processing();

  // Create Task for OpenMP
  kostanyan_omp_sobel::Kostanyan_EdgeDetectionParallel kostanyan_EdgeDetectionParallel(taskDataOmp);
  ASSERT_EQ(kostanyan_EdgeDetectionParallel.validation(), true);
  kostanyan_EdgeDetectionParallel.pre_processing();
  kostanyan_EdgeDetectionParallel.run();
  kostanyan_EdgeDetectionParallel.post_processing();

  // Check result
  for (int i = 1; i < n - 1; i++) {
    for (int j = 1; j < m - 1; j++) {
      ASSERT_EQ(out_seq[i * m + j], out_omp[i * m + j]);
    }
  }
}

TEST(kostanyan_a_sobel_omp, Test_Empty_Image) {
  const int n = 0;
  const int m = 0;

  // Create data
  std::vector<int> in(2);
  in[0] = n;
  in[1] = m;
  std::vector<uint8_t> pict;
  std::vector<uint8_t> out_seq;
  std::vector<uint8_t> out_omp;

  // Create TaskData for Sequential
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(pict.data());
  taskDataSeq->inputs_count.emplace_back(pict.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  // Create TaskData for OpenMP
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataOmp->inputs_count.emplace_back(in.size());
  taskDataOmp->inputs.emplace_back(pict.data());
  taskDataOmp->inputs_count.emplace_back(pict.size());
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataOmp->outputs_count.emplace_back(out_omp.size());

  // Create Task for Sequential
  kostanyan_omp_sobel::Kostanyan_EdgeDetectionSequential kostanyan_EdgeDetectionSequential(taskDataSeq);
  ASSERT_EQ(kostanyan_EdgeDetectionSequential.validation(), true);
  kostanyan_EdgeDetectionSequential.pre_processing();
  ASSERT_EQ(kostanyan_EdgeDetectionSequential.run(), false);

  // Create Task for OpenMP
  kostanyan_omp_sobel::Kostanyan_EdgeDetectionParallel kostanyan_EdgeDetectionParallel(taskDataOmp);
  ASSERT_EQ(kostanyan_EdgeDetectionParallel.validation(), true);
  kostanyan_EdgeDetectionParallel.pre_processing();
  ASSERT_EQ(kostanyan_EdgeDetectionParallel.run(), false);
}

TEST(kostanyan_a_sobel_omp, Test_Single_Pixel_Image) {
  const int n = 1;
  const int m = 1;
  const uint8_t min = 0;
  const uint8_t max = 255;

  // Create data
  std::vector<int> in(2);
  in[0] = n;
  in[1] = m;
  std::vector<uint8_t> pict = kostanyan_omp_sobel::kostanyan_getRandomPicture(n, m, min, max);
  std::vector<uint8_t> out_seq(n * m, 0);
  std::vector<uint8_t> out_omp(n * m, 0);

  // Create TaskData for Sequential
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(pict.data());
  taskDataSeq->inputs_count.emplace_back(pict.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  // Create TaskData for OpenMP
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataOmp->inputs_count.emplace_back(in.size());
  taskDataOmp->inputs.emplace_back(pict.data());
  taskDataOmp->inputs_count.emplace_back(pict.size());
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataOmp->outputs_count.emplace_back(out_omp.size());

  // Create Task for Sequential
  kostanyan_omp_sobel::Kostanyan_EdgeDetectionSequential kostanyan_EdgeDetectionSequential(taskDataSeq);
  ASSERT_EQ(kostanyan_EdgeDetectionSequential.validation(), true);
  kostanyan_EdgeDetectionSequential.pre_processing();
  kostanyan_EdgeDetectionSequential.run();
  kostanyan_EdgeDetectionSequential.post_processing();
  ASSERT_EQ(pict[0], out_seq[0]);

  // Create Task for OpenMP
  kostanyan_omp_sobel::Kostanyan_EdgeDetectionParallel kostanyan_EdgeDetectionParallel(taskDataOmp);
  ASSERT_EQ(kostanyan_EdgeDetectionParallel.validation(), true);
  kostanyan_EdgeDetectionParallel.pre_processing();
  kostanyan_EdgeDetectionParallel.run();
  kostanyan_EdgeDetectionParallel.post_processing();
  ASSERT_EQ(pict[0], out_omp[0]);
}

TEST(kostanyan_a_sobel_omp, Test_All_White_Image) {
  const int n = 10;
  const int m = 10;
  const uint8_t white = 255;

  // Create data
  std::vector<int> in(2);
  in[0] = n;
  in[1] = m;
  std::vector<uint8_t> pict(n * m, white);
  std::vector<uint8_t> out_seq(n * m, 0);
  std::vector<uint8_t> out_omp(n * m, 0);

  // Create TaskData for Sequential
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(pict.data());
  taskDataSeq->inputs_count.emplace_back(pict.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  // Create TaskData for OpenMP
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataOmp->inputs_count.emplace_back(in.size());
  taskDataOmp->inputs.emplace_back(pict.data());
  taskDataOmp->inputs_count.emplace_back(pict.size());
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataOmp->outputs_count.emplace_back(out_omp.size());

  // Create Task for Sequential
  kostanyan_omp_sobel::Kostanyan_EdgeDetectionSequential kostanyan_EdgeDetectionSequential(taskDataSeq);
  ASSERT_EQ(kostanyan_EdgeDetectionSequential.validation(), true);
  kostanyan_EdgeDetectionSequential.pre_processing();
  kostanyan_EdgeDetectionSequential.run();
  kostanyan_EdgeDetectionSequential.post_processing();

  // Create Task for OpenMP
  kostanyan_omp_sobel::Kostanyan_EdgeDetectionParallel kostanyan_EdgeDetectionParallel(taskDataOmp);
  ASSERT_EQ(kostanyan_EdgeDetectionParallel.validation(), true);
  kostanyan_EdgeDetectionParallel.pre_processing();
  kostanyan_EdgeDetectionParallel.run();
  kostanyan_EdgeDetectionParallel.post_processing();

  for (int i = 1; i < n - 1; i++) {
    for (int j = 1; j < m - 1; j++) {
      ASSERT_EQ(out_seq[i * m + j], out_omp[i * m + j]);
    }
  }
}

TEST(kostanyan_a_sobel_omp, Test_Large_Image) {
  const int n = 100;
  const int m = 100;
  const uint8_t min = 0;
  const uint8_t max = 255;

  // Create data
  std::vector<int> in(2);
  in[0] = n;
  in[1] = m;
  std::vector<uint8_t> pict = kostanyan_omp_sobel::kostanyan_getRandomPicture(n, m, min, max);
  std::vector<uint8_t> out_seq(n * m, 0);
  std::vector<uint8_t> out_omp(n * m, 0);

  // Create TaskData for Sequential
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(pict.data());
  taskDataSeq->inputs_count.emplace_back(pict.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
  taskDataSeq->outputs_count.emplace_back(out_seq.size());

  // Create TaskData for OpenMP
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataOmp->inputs_count.emplace_back(in.size());
  taskDataOmp->inputs.emplace_back(pict.data());
  taskDataOmp->inputs_count.emplace_back(pict.size());
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_omp.data()));
  taskDataOmp->outputs_count.emplace_back(out_omp.size());

  // Create Task for Sequential
  kostanyan_omp_sobel::Kostanyan_EdgeDetectionSequential kostanyan_EdgeDetectionSequential(taskDataSeq);
  ASSERT_EQ(kostanyan_EdgeDetectionSequential.validation(), true);
  kostanyan_EdgeDetectionSequential.pre_processing();
  kostanyan_EdgeDetectionSequential.run();
  kostanyan_EdgeDetectionSequential.post_processing();

  // Create Task for OpenMP
  kostanyan_omp_sobel::Kostanyan_EdgeDetectionParallel kostanyan_EdgeDetectionParallel(taskDataOmp);
  ASSERT_EQ(kostanyan_EdgeDetectionParallel.validation(), true);
  kostanyan_EdgeDetectionParallel.pre_processing();
  kostanyan_EdgeDetectionParallel.run();
  kostanyan_EdgeDetectionParallel.post_processing();

  for (int i = 1; i < n - 1; i++) {
    for (int j = 1; j < m - 1; j++) {
      ASSERT_EQ(out_seq[i * m + j], out_omp[i * m + j]);
    }
  }
}
