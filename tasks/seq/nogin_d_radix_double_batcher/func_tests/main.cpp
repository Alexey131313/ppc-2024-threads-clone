// Copyright 2024 Nogin Denis
#include <gtest/gtest.h>

#include "seq/nogin_d_radix_double_batcher/include/ops_seq.hpp"

TEST(nogin_d_radix_double_batcher_seq, array) {
  // Create data
  std::vector<double> array = {4.44, 8.88, 1.11, 2.22, 9.99, 6.66, 5.55, 3.33, 7.77};
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
  NoginDenisSeq::RadixSortDoubleBatcherSequential testRDSBSequential(RSDBDataSeq);
  ASSERT_EQ(testRDSBSequential.validation(), true);
  testRDSBSequential.pre_processing();
  testRDSBSequential.run();
  testRDSBSequential.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(nogin_d_radix_double_batcher_seq, test_array_2) {
  // Create data
  std::vector<double> array = {67.1234, 29.567,  83.9876, 14.70,   95.345,  42.245,  55.678,  78.3456, 6.801,
                               37.8765, 20.4567, 51.245,  71.8765, 1.3456,  63.2345, 88.8,    46.7890, 22.1234,
                               5.89,    8.3456,  33.7890, 17.2345, 91.5678, 59.1234, 25.6789, 69.346,  8.7890,
                               40.1234, 80.5678, 52.3456, 16.7890, 93.2345, 74.5678, 38.12,   9.356,   31.7890,
                               86.235,  12.5678, 64.123,  44.5678, 77.2345, 35.5678, 89.7890, 68.234,  26.5678,
                               56.234,  3.5678,  70.7890, 18.134,  49.5678};
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
  NoginDenisSeq::RadixSortDoubleBatcherSequential testRDSBSequential(RSDBDataSeq);
  ASSERT_EQ(testRDSBSequential.validation(), true);
  testRDSBSequential.pre_processing();
  testRDSBSequential.run();
  testRDSBSequential.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(nogin_d_radix_double_batcher_seq, test_empty) {
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
  NoginDenisSeq::RadixSortDoubleBatcherSequential testRDSBSequential(RSDBDataSeq);
  ASSERT_EQ(testRDSBSequential.validation(), true);
  testRDSBSequential.pre_processing();
  testRDSBSequential.run();
  testRDSBSequential.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(nogin_d_radix_double_batcher_seq, test_one_number) {
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
  NoginDenisSeq::RadixSortDoubleBatcherSequential testRDSBSequential(RSDBDataSeq);
  ASSERT_EQ(testRDSBSequential.validation(), true);
  testRDSBSequential.pre_processing();
  testRDSBSequential.run();
  testRDSBSequential.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(nogin_d_radix_double_batcher_seq, test_with_negative) {
  // Create data
  std::vector<double> array = {1.1, -11.1, 5.5, 5, 5.3, 9.9, -6.6, 0, 3.3, 0, -1.1, 15, -2};
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
  NoginDenisSeq::RadixSortDoubleBatcherSequential testRDSBSequential(RSDBDataSeq);
  ASSERT_EQ(testRDSBSequential.validation(), true);
  testRDSBSequential.pre_processing();
  testRDSBSequential.run();
  testRDSBSequential.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}

TEST(nogin_d_radix_double_batcher_seq, test_random) {
  // Create data
  std::vector<double> array = NoginDenisSeq::randomVector(1000, -1000000, 1000000);
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
  NoginDenisSeq::RadixSortDoubleBatcherSequential testRDSBSequential(RSDBDataSeq);
  ASSERT_EQ(testRDSBSequential.validation(), true);
  testRDSBSequential.pre_processing();
  testRDSBSequential.run();
  testRDSBSequential.post_processing();
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], out[i]);
  }
}
