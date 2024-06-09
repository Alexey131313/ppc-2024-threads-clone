// Copyright 2024 Ulyanov Daniil
#include <gtest/gtest.h>

#include <vector>

#include "omp/ulyanov_daniil/include/ops_omp.hpp"

TEST(Par_OMP_Ulyanov, Test1_Ulyanov) {
  int height = 100;
  int width = 100;

  // Create data
  std::vector<uint8_t> in(height * width * 3);
  for (size_t i = 0; i < in.size(); i++) {
    in[i] = i % 255;
  }
  std::vector<uint8_t> out_seq(height * width * 3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(in.data());
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(out_seq.data());
  taskDataSeq->outputs_count.emplace_back(height);
  taskDataSeq->outputs_count.emplace_back(width);

  // Create Task
  OMPFilterGaussHorizontalSeqUlyanov testOmpTaskSequential(taskDataSeq);
  ASSERT_EQ(testOmpTaskSequential.validation(), true);
  testOmpTaskSequential.pre_processing();
  testOmpTaskSequential.run();
  testOmpTaskSequential.post_processing();

  // Create data
  std::vector<uint8_t> out_par(height * width * 3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(in.data());
  taskDataPar->inputs_count.emplace_back(height);
  taskDataPar->inputs_count.emplace_back(width);
  taskDataPar->outputs.emplace_back(out_par.data());
  taskDataPar->outputs_count.emplace_back(height);
  taskDataPar->outputs_count.emplace_back(width);

  // Create Task
  OMPFilterGaussHorizontalParUlyanov testOmpTaskParallel(taskDataPar);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  ASSERT_EQ(out_seq, out_par);
}

TEST(Par_OMP_Ulyanov, Test2_Ulyanov) {
  int height = 500;
  int width = 500;

  // Create data
  std::vector<uint8_t> in(height * width * 3);
  for (size_t i = 0; i < in.size(); i++) {
    in[i] = i % 255;
  }
  std::vector<uint8_t> out_seq(height * width * 3);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(in.data());
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(out_seq.data());
  taskDataSeq->outputs_count.emplace_back(height);
  taskDataSeq->outputs_count.emplace_back(width);

  // Create Task
  OMPFilterGaussHorizontalSeqUlyanov testOmpTaskSequential(taskDataSeq);
  ASSERT_EQ(testOmpTaskSequential.validation(), true);
  testOmpTaskSequential.pre_processing();
  testOmpTaskSequential.run();
  testOmpTaskSequential.post_processing();

  // Create data
  std::vector<uint8_t> out_par(height * width * 3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(in.data());
  taskDataPar->inputs_count.emplace_back(height);
  taskDataPar->inputs_count.emplace_back(width);
  taskDataPar->outputs.emplace_back(out_par.data());
  taskDataPar->outputs_count.emplace_back(height);
  taskDataPar->outputs_count.emplace_back(width);

  // Create Task
  OMPFilterGaussHorizontalParUlyanov testOmpTaskParallel(taskDataPar);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  ASSERT_EQ(out_seq, out_par);
}

TEST(Par_OMP_Ulyanov, Test3_Ulyanov) {
  int height = 200;
  int width = 200;

  // Create data
  std::vector<uint8_t> in(height * width * 3);
  for (size_t i = 0; i < in.size(); i++) {
    in[i] = (i + 100) % 255;
  }
  std::vector<uint8_t> out_seq(height * width * 3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(in.data());
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(out_seq.data());
  taskDataSeq->outputs_count.emplace_back(height);
  taskDataSeq->outputs_count.emplace_back(width);

  // Create Data
  std::vector<uint8_t> out_par(height * width * 3);

  // Create Task
  OMPFilterGaussHorizontalSeqUlyanov testOmpTaskSequential(taskDataSeq);
  ASSERT_EQ(testOmpTaskSequential.validation(), true);
  testOmpTaskSequential.pre_processing();
  testOmpTaskSequential.run();
  testOmpTaskSequential.post_processing();

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(in.data());
  taskDataPar->inputs_count.emplace_back(height);
  taskDataPar->inputs_count.emplace_back(width);
  taskDataPar->outputs.emplace_back(out_par.data());
  taskDataPar->outputs_count.emplace_back(height);
  taskDataPar->outputs_count.emplace_back(width);

  // Create Task
  OMPFilterGaussHorizontalParUlyanov testOmpTaskParallel(taskDataPar);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  ASSERT_EQ(out_seq, out_par);
}

TEST(Par_OMP_Ulyanov, Test4_Ulyanov) {
  int height = 123;
  int width = 123;

  // Create data
  std::vector<uint8_t> in(height * width * 3);
  for (size_t i = 0; i < in.size(); i++) {
    in[i] = i % 255;
  }
  std::vector<uint8_t> out_seq(height * width * 3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(in.data());
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(out_seq.data());
  taskDataSeq->outputs_count.emplace_back(height);
  taskDataSeq->outputs_count.emplace_back(width);

  // Create Task
  OMPFilterGaussHorizontalSeqUlyanov testOmpTaskSequential(taskDataSeq);
  ASSERT_EQ(testOmpTaskSequential.validation(), true);
  testOmpTaskSequential.pre_processing();
  testOmpTaskSequential.run();
  testOmpTaskSequential.post_processing();

  // Create data
  std::vector<uint8_t> out_par(height * width * 3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(in.data());
  taskDataPar->inputs_count.emplace_back(height);
  taskDataPar->inputs_count.emplace_back(width);
  taskDataPar->outputs.emplace_back(out_par.data());
  taskDataPar->outputs_count.emplace_back(height);
  taskDataPar->outputs_count.emplace_back(width);

  // Create Task
  OMPFilterGaussHorizontalParUlyanov testOmpTaskParallel(taskDataPar);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  ASSERT_EQ(out_seq, out_par);
}

TEST(Par_OMP_Ulyanov, Test5_Ulyanov) {
  int height = 10;
  int width = 10;

  // Create data
  std::vector<uint8_t> in(height * width * 3);
  for (size_t i = 0; i < in.size(); i++) {
    in[i] = i % 255;
  }
  std::vector<uint8_t> out_seq(height * width * 3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(in.data());
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(out_seq.data());
  taskDataSeq->outputs_count.emplace_back(height);
  taskDataSeq->outputs_count.emplace_back(width);

  // Create Task
  OMPFilterGaussHorizontalSeqUlyanov testOmpTaskSequential(taskDataSeq);
  ASSERT_EQ(testOmpTaskSequential.validation(), true);
  testOmpTaskSequential.pre_processing();
  testOmpTaskSequential.run();
  testOmpTaskSequential.post_processing();

  // Create data
  std::vector<uint8_t> out_par(height * width * 3);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(in.data());
  taskDataPar->inputs_count.emplace_back(height);
  taskDataPar->inputs_count.emplace_back(width);
  taskDataPar->outputs.emplace_back(out_par.data());
  taskDataPar->outputs_count.emplace_back(height);
  taskDataPar->outputs_count.emplace_back(width);

  // Create Task
  OMPFilterGaussHorizontalParUlyanov testOmpTaskParallel(taskDataPar);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  ASSERT_EQ(out_seq, out_par);
}

// int main(int argc, char **argv) {
//   ::testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
