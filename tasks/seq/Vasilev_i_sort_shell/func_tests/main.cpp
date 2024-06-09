// Copyright 2024 Vasilev Ivan
#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "seq/Vasilev_i_sort_shell/include/ops_seq.hpp"

// TEST(Sequential, Test_Sum_10) {
//   const int count = 10;

//   // Create data
//   std::vector<int> in(1, count);
//   std::vector<int> out(1, 0);

//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   taskDataSeq->inputs_count.emplace_back(in.size());
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());

//   // Create Task
//   TestTaskSequential testTaskSequential(taskDataSeq);
//   ASSERT_EQ(testTaskSequential.validation(), true);
//   testTaskSequential.pre_processing();
//   testTaskSequential.run();
//   testTaskSequential.post_processing();
//   ASSERT_EQ(count, out[0]);
// }

// TEST(Sequential, Test_Sum_20) {
//   const int count = 20;

//   // Create data
//   std::vector<int> in(1, count);
//   std::vector<int> out(1, 0);

//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   taskDataSeq->inputs_count.emplace_back(in.size());
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());

//   // Create Task
//   TestTaskSequential testTaskSequential(taskDataSeq);
//   ASSERT_EQ(testTaskSequential.validation(), true);
//   testTaskSequential.pre_processing();
//   testTaskSequential.run();
//   testTaskSequential.post_processing();
//   ASSERT_EQ(count, out[0]);
// }

// TEST(Sequential, Test_Sum_50) {
//   const int count = 50;

//   // Create data
//   std::vector<int> in(1, count);
//   std::vector<int> out(1, 0);

//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   taskDataSeq->inputs_count.emplace_back(in.size());
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());

//   // Create Task
//   TestTaskSequential testTaskSequential(taskDataSeq);
//   ASSERT_EQ(testTaskSequential.validation(), true);
//   testTaskSequential.pre_processing();
//   testTaskSequential.run();
//   testTaskSequential.post_processing();
//   ASSERT_EQ(count, out[0]);
// }

// TEST(Sequential, Test_Sum_70) {
//   const int count = 70;

//   // Create data
//   std::vector<int> in(1, count);
//   std::vector<int> out(1, 0);

//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   taskDataSeq->inputs_count.emplace_back(in.size());
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());

//   // Create Task
//   TestTaskSequential testTaskSequential(taskDataSeq);
//   ASSERT_EQ(testTaskSequential.validation(), true);
//   testTaskSequential.pre_processing();
//   testTaskSequential.run();
//   testTaskSequential.post_processing();
//   ASSERT_EQ(count, out[0]);
// }

// TEST(Sequential, Test_Sum_100) {
//   const int count = 100;

//   // Create data
//   std::vector<int> in(1, count);
//   std::vector<int> out(1, 0);

//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   taskDataSeq->inputs_count.emplace_back(in.size());
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());

//   // Create Task
//   TestTaskSequential testTaskSequential(taskDataSeq);
//   ASSERT_EQ(testTaskSequential.validation(), true);
//   testTaskSequential.pre_processing();
//   testTaskSequential.run();
//   testTaskSequential.post_processing();
//   ASSERT_EQ(count, out[0]);
// }

TEST(vasilev_i_sort_shell_seq, Sort_2) {
  std::vector<int> arr = {2, 1};
  std::vector<int> rez(2);
  std::vector<int> expected = {1, 2};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&arr));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&rez));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  VasilevTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(rez, expected);
}

TEST(vasilev_i_sort_shell_seq, Sort_4) {
  std::vector<int> arr = {3, 2, 4, 1};
  std::vector<int> rez(4);
  std::vector<int> expected = {1, 2, 3, 4};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&arr));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&rez));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  VasilevTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(rez, expected);
}

TEST(vasilev_i_sort_shell_seq, Sort_8) {
  std::vector<int> arr = {5, 3, 8, 6, 2, 7, 1, 4};
  std::vector<int> rez(8);
  std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&arr));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&rez));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  VasilevTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(rez, expected);
}

TEST(vasilev_i_sort_shell_seq, Sort_16) {
  std::vector<int> arr = {5, 8, 11, 6, 12, 16, 9, 3, 13, 2, 7, 4, 10, 1, 14, 15};
  std::vector<int> rez(16);
  std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&arr));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&rez));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  VasilevTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(rez, expected);
}

TEST(vasilev_i_sort_shell_seq, Sort_32) {
  std::vector<int> arr = {3,  24, 15, 10, 32, 1, 25, 30, 8,  14, 28, 27, 2,  21, 6,  16,
                          20, 26, 12, 13, 18, 5, 4,  23, 11, 29, 19, 9,  22, 17, 31, 7};
  std::vector<int> rez(32);
  std::vector<int> expected = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                               17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&arr));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&rez));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  VasilevTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(rez, expected);
}
// int main(int argc, char **argv) {
//   testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
