// Copyright Lebedeva_Ksenia 2024
#include <gtest/gtest.h>

#include "seq/lebedeva_k_bitwise_sort_double_batcher/include/ops_seq.hpp"

TEST(lebedeva_k_bitwise_sort_double_batcher, array) {
  // Create data
  std::vector<double> array = {-12.346, 45.678, -89.012, 34.567, 90.023, -56.789, 23.456, -78.901, 67.890};
  std::vector<double> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> RSDBDataSeq = std::make_shared<ppc::core::TaskData>();
  RSDBDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  RSDBDataSeq->inputs_count.emplace_back(array.size());
  RSDBDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  RSDBDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  RadixSortDoubleBatcherSequential testRDSBSequential(RSDBDataSeq);
  ASSERT_EQ(testRDSBSequential.validation(), true);
  testRDSBSequential.pre_processing();
  testRDSBSequential.run();
  testRDSBSequential.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(lebedeva_k_bitwise_sort_double_batcher, test_array_2) {
  // Create data
  std::vector<double> array = {72.3352,  -89.4352, 15.1976,  38.5129,  5.24859,  -17.2231, -34.1062, 22.1978,  1.75238,
                               -64.1476, -41.2428, 68.8762,  -78.4872, 70.6463,  60.0645,  -59.8497, -96.6727, -24.6936,
                               39.1572,  -73.3399, 94.1972,  -66.3034, -18.5411, -99.2365, -95.9891, 14.0173,  4.49572,
                               -64.0323, -54.2447, 49.4073,  -48.8544, -18.1294, -27.0456, 60.0547,  92.1093,  -81.1795,
                               -45.1096, -45.2323, 33.5223,  32.2206,  -15.7836, -99.5654, 39.6649,  47.8456,  85.2707,
                               81.4374,  -92.8531, -38.3659, 81.9927,  -4.1559};
  std::vector<double> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> RSDBDataSeq = std::make_shared<ppc::core::TaskData>();
  RSDBDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  RSDBDataSeq->inputs_count.emplace_back(array.size());
  RSDBDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  RSDBDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  RadixSortDoubleBatcherSequential testRDSBSequential(RSDBDataSeq);
  ASSERT_EQ(testRDSBSequential.validation(), true);
  testRDSBSequential.pre_processing();
  testRDSBSequential.run();
  testRDSBSequential.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(lebedeva_k_bitwise_sort_double_batcher, test_empty) {
  // Create data
  std::vector<double> array = {};
  std::vector<double> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> RSDBDataSeq = std::make_shared<ppc::core::TaskData>();
  RSDBDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  RSDBDataSeq->inputs_count.emplace_back(array.size());
  RSDBDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  RSDBDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  RadixSortDoubleBatcherSequential testRDSBSequential(RSDBDataSeq);
  ASSERT_EQ(testRDSBSequential.validation(), true);
  testRDSBSequential.pre_processing();
  testRDSBSequential.run();
  testRDSBSequential.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(lebedeva_k_bitwise_sort_double_batcher, test_one_number) {
  // Create data
  std::vector<double> array = {1};
  std::vector<double> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> RSDBDataSeq = std::make_shared<ppc::core::TaskData>();
  RSDBDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  RSDBDataSeq->inputs_count.emplace_back(array.size());
  RSDBDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  RSDBDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  RadixSortDoubleBatcherSequential testRDSBSequential(RSDBDataSeq);
  ASSERT_EQ(testRDSBSequential.validation(), true);
  testRDSBSequential.pre_processing();
  testRDSBSequential.run();
  testRDSBSequential.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(lebedeva_k_bitwise_sort_double_batcher, test_with_negative) {
  // Create data
  std::vector<double> array = {-45.678, 12.345, 90.123,  -23.456, 67.890,  -34.567, 56.789,
                               -78.901, 89.012, -56.789, 45.678,  -90.123, 34.567};
  std::vector<double> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> RSDBDataSeq = std::make_shared<ppc::core::TaskData>();
  RSDBDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  RSDBDataSeq->inputs_count.emplace_back(array.size());
  RSDBDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  RSDBDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  RadixSortDoubleBatcherSequential testRDSBSequential(RSDBDataSeq);
  ASSERT_EQ(testRDSBSequential.validation(), true);
  testRDSBSequential.pre_processing();
  testRDSBSequential.run();
  testRDSBSequential.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(lebedeva_k_bitwise_sort_double_batcher, test_random) {
  // Create data
  std::vector<double> array = randomVector(1000, -1000000, 1000000);
  std::vector<double> out(array.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> RSDBDataSeq = std::make_shared<ppc::core::TaskData>();
  RSDBDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(array.data()));
  RSDBDataSeq->inputs_count.emplace_back(array.size());
  RSDBDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  RSDBDataSeq->outputs_count.emplace_back(out.size());

  std::vector<double> result(array);
  std::sort(result.begin(), result.end());

  // Create Task
  RadixSortDoubleBatcherSequential testRDSBSequential(RSDBDataSeq);
  ASSERT_EQ(testRDSBSequential.validation(), true);
  testRDSBSequential.pre_processing();
  testRDSBSequential.run();
  testRDSBSequential.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}
