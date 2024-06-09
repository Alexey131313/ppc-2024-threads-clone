// Copyright 2024 Savchuk Anton
#include <gtest/gtest.h>

#include <vector>

#include "tbb/savchuk_a_crs_matmult_tbb/include/crs_matmult_tbb.hpp"

using namespace SavchukTbb;

TEST(savchuk_a_crs_matmult_tbb, test_sizes) {
  size_t n1 = 4;
  size_t m1 = 6;
  size_t n2 = 6;
  size_t m2 = 2;

  // Create data
  std::vector<Complex> in1(n1 * m1);

  std::vector<Complex> in2(n2 * m2);

  std::vector<Complex> out(n1 * m2);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(n1);
  taskDataSeq->inputs_count.emplace_back(m1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(n2);
  taskDataSeq->inputs_count.emplace_back(m2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n1);
  taskDataSeq->outputs_count.emplace_back(m2);

  // Create Task
  SavchukCRSMatMultTBBSequential savchukCRSMatMultTBBSequential(taskDataSeq);
  ASSERT_EQ(savchukCRSMatMultTBBSequential.validation(), true);

  // Create data
  std::vector<Complex> in3(n1 * m1);

  std::vector<Complex> in4(n2 * m2);

  std::vector<Complex> out2(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataParallel->inputs_count.emplace_back(n1);
  taskDataParallel->inputs_count.emplace_back(m1);
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in4.data()));
  taskDataParallel->inputs_count.emplace_back(n2);
  taskDataParallel->inputs_count.emplace_back(m2);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataParallel->outputs_count.emplace_back(n1);
  taskDataParallel->outputs_count.emplace_back(m2);
  // Create Task
  SavchukCRSMatMultTBBParallel savchukCRSMatMultTBBParallel(taskDataParallel);
  ASSERT_EQ(savchukCRSMatMultTBBParallel.validation(), true);
}

TEST(savchuk_a_crs_matmult_tbb, test_sizes2) {
  size_t n1 = 2;
  size_t m1 = 3;
  size_t n2 = 4;
  size_t m2 = 5;

  // Create data
  std::vector<Complex> in1(n1 * m1);

  std::vector<Complex> in2(n2 * m2);

  std::vector<Complex> out(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(n1);
  taskDataSeq->inputs_count.emplace_back(m1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(n2);
  taskDataSeq->inputs_count.emplace_back(m2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n1);
  taskDataSeq->outputs_count.emplace_back(m2);

  // Create Task
  SavchukCRSMatMultTBBSequential savchukCRSMatMultTBBSequential(taskDataSeq);
  ASSERT_FALSE(savchukCRSMatMultTBBSequential.validation());

  // Create data
  std::vector<Complex> in3(n1 * m1);

  std::vector<Complex> in4(n2 * m2);

  std::vector<Complex> out2(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in3.data()));
  taskDataParallel->inputs_count.emplace_back(n1);
  taskDataParallel->inputs_count.emplace_back(m1);
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in4.data()));
  taskDataParallel->inputs_count.emplace_back(n2);
  taskDataParallel->inputs_count.emplace_back(m2);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataParallel->outputs_count.emplace_back(n1);
  taskDataParallel->outputs_count.emplace_back(m2);
  // Create Task
  SavchukCRSMatMultTBBParallel savchukCRSMatMultTBBParallel(taskDataParallel);
  ASSERT_FALSE(savchukCRSMatMultTBBParallel.validation());
}

TEST(savchuk_a_crs_matmult_tbb, test_multy_correct) {
  size_t n1 = 3;
  size_t m1 = 3;
  size_t n2 = 3;
  size_t m2 = 3;
  // Create data
  std::vector<Complex> in1{Complex(3, 2), Complex(0, 0), Complex(0, 0), Complex(0, 0), Complex(1, -3),
                           Complex(0, 0), Complex(0, 0), Complex(0, 0), Complex(-4, 1)};
  std::vector<Complex> in2{Complex(0, 0),  Complex(2, -1), Complex(0, 0), Complex(0, 0), Complex(0, 0),
                           Complex(-5, 2), Complex(0, 0),  Complex(2, 1), Complex(0, 0)};
  std::vector<Complex> out(n1 * m2);
  std::vector<Complex> test{Complex(0, 0),  Complex(8, 1), Complex(0, 0),   Complex(0, 0), Complex(0, 0),
                            Complex(1, 17), Complex(0, 0), Complex(-9, -2), Complex(0, 0)};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(n1);
  taskDataSeq->inputs_count.emplace_back(m1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(n2);
  taskDataSeq->inputs_count.emplace_back(m2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n1);
  taskDataSeq->outputs_count.emplace_back(m2);

  // Create Task
  SavchukCRSMatMultTBBSequential savchukCRSMatMultTBBSequential(taskDataSeq);
  savchukCRSMatMultTBBSequential.validation();
  savchukCRSMatMultTBBSequential.pre_processing();
  savchukCRSMatMultTBBSequential.run();
  savchukCRSMatMultTBBSequential.post_processing();

  size_t k = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == test[i]) {
      k++;
    }
  }

  ASSERT_EQ(k, n1 * m2);
  // Create data
  std::vector<Complex> out2(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataParallel->inputs_count.emplace_back(n1);
  taskDataParallel->inputs_count.emplace_back(m1);
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataParallel->inputs_count.emplace_back(n2);
  taskDataParallel->inputs_count.emplace_back(m2);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataParallel->outputs_count.emplace_back(n1);
  taskDataParallel->outputs_count.emplace_back(m2);

  // Create Task
  SavchukCRSMatMultTBBParallel savchukCRSMatMultTBBParallel(taskDataParallel);
  savchukCRSMatMultTBBParallel.validation();
  savchukCRSMatMultTBBParallel.pre_processing();
  savchukCRSMatMultTBBParallel.run();
  savchukCRSMatMultTBBParallel.post_processing();

  size_t ch = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == test[i]) {
      ch++;
    }
  }

  ASSERT_EQ(ch, n1 * m2);
}

TEST(savchuk_a_crs_matmult_tbb, inverse_matrix) {
  size_t n1 = 3;
  size_t m1 = 3;
  size_t n2 = 3;
  size_t m2 = 3;

  // Create data
  std::vector<Complex> in1{Complex(2, -1), Complex(0, 0), Complex(0, 0), Complex(0, 0), Complex(0, 0),
                           Complex(0, 2),  Complex(0, 0), Complex(2, 1), Complex(0, 0)};
  std::vector<Complex> in2{Complex(0.4, 0.2),  Complex(0, 0), Complex(0, 0),    Complex(0, 0), Complex(0, 0),
                           Complex(0.4, -0.2), Complex(0, 0), Complex(0, -0.5), Complex(0, 0)};
  std::vector<Complex> out(n1 * m2);
  std::vector<Complex> identity{Complex(1, 0), Complex(0, 0), Complex(0, 0), Complex(0, 0), Complex(1, 0),
                                Complex(0, 0), Complex(0, 0), Complex(0, 0), Complex(1, 0)};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(n1);
  taskDataSeq->inputs_count.emplace_back(m1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(n2);
  taskDataSeq->inputs_count.emplace_back(m2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n1);
  taskDataSeq->outputs_count.emplace_back(m2);

  // Create Task
  SavchukCRSMatMultTBBSequential savchukCRSMatMultTBBSequential(taskDataSeq);
  savchukCRSMatMultTBBSequential.validation();
  savchukCRSMatMultTBBSequential.pre_processing();
  savchukCRSMatMultTBBSequential.run();
  savchukCRSMatMultTBBSequential.post_processing();

  size_t k = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == identity[i]) {
      k++;
    }
  }

  ASSERT_EQ(k, n1 * m2);
  // Create data
  std::vector<Complex> out2(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataParallel->inputs_count.emplace_back(n1);
  taskDataParallel->inputs_count.emplace_back(m1);
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataParallel->inputs_count.emplace_back(n2);
  taskDataParallel->inputs_count.emplace_back(m2);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataParallel->outputs_count.emplace_back(n1);
  taskDataParallel->outputs_count.emplace_back(m2);

  // Create Task
  SavchukCRSMatMultTBBParallel savchukCRSMatMultTBBParallel(taskDataSeq);
  savchukCRSMatMultTBBParallel.validation();
  savchukCRSMatMultTBBParallel.pre_processing();
  savchukCRSMatMultTBBParallel.run();
  savchukCRSMatMultTBBParallel.post_processing();

  size_t ch = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == identity[i]) {
      ch++;
    }
  }

  ASSERT_EQ(ch, n1 * m2);
}

TEST(savchuk_a_crs_matmult_tbb, zero_matrix) {
  size_t n1 = 3;
  size_t m1 = 3;
  size_t n2 = 3;
  size_t m2 = 3;
  // Create data
  Complex Null = Complex(0, 0);
  std::vector<Complex> in1{Complex(7, 1), Complex(0, 0), Complex(0, 0), Complex(2, -6), Complex(0, 5),
                           Complex(0, 0), Complex(1, 0), Complex(0, 0), Complex(0, 0)};
  std::vector<Complex> in2{Null, Null, Null, Null, Null, Null, Null, Null, Null};
  std::vector<Complex> out(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataSeq->inputs_count.emplace_back(n1);
  taskDataSeq->inputs_count.emplace_back(m1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataSeq->inputs_count.emplace_back(n2);
  taskDataSeq->inputs_count.emplace_back(m2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n1);
  taskDataSeq->outputs_count.emplace_back(m2);

  // Create Task
  SavchukCRSMatMultTBBSequential savchukCRSMatMultTBBSequential(taskDataSeq);
  savchukCRSMatMultTBBSequential.validation();
  savchukCRSMatMultTBBSequential.pre_processing();
  savchukCRSMatMultTBBSequential.run();
  savchukCRSMatMultTBBSequential.post_processing();

  size_t k = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == Complex(0.0, 0.0)) {
      k++;
    }
  }

  ASSERT_EQ(k, n1 * m2);
  // Create data
  std::vector<Complex> out2(n1 * m2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  taskDataParallel->inputs_count.emplace_back(n1);
  taskDataParallel->inputs_count.emplace_back(m1);
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  taskDataParallel->inputs_count.emplace_back(n2);
  taskDataParallel->inputs_count.emplace_back(m2);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out2.data()));
  taskDataParallel->outputs_count.emplace_back(n1);
  taskDataParallel->outputs_count.emplace_back(m2);

  // Create Task
  SavchukCRSMatMultTBBParallel savchukCRSMatMultTBBParallel(taskDataSeq);
  savchukCRSMatMultTBBParallel.validation();
  savchukCRSMatMultTBBParallel.pre_processing();
  savchukCRSMatMultTBBParallel.run();
  savchukCRSMatMultTBBParallel.post_processing();

  size_t ch = 0;

  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == Complex(0.0, 0.0)) {
      ch++;
    }
  }

  ASSERT_EQ(ch, n1 * m2);
}