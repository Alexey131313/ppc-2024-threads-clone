// Copyright 2024 Mortina Nastya
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "tbb/mortina_a_int_trapezoid_tbb/include/ops_tbb.hpp"

TEST(Parallel_mortina_a_int_trapizoid_TBB, Test_1) {
  double a1 = 0.0;
  double b1 = 2.0;
  double a2 = 0.0;
  double b2 = 2.0;

  int n1 = 30;
  int n2 = 30;

  double res = 7.11;

  auto x_y_quadro = [](double x, double y) { return (x * x) * (y * y); };

  // Create data
  std::vector<double> in = {a1, b1, a2, b2};
  std::vector<double> out(1, res);
  std::vector<int> n = {n1, n2};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  mortina_a_integral_trapezoid_tbb::TestTBBTaskSequentialMortinaIntegralTrapezoid testOmpTaskSequential(taskDataSeq,
                                                                                                        x_y_quadro);
  ASSERT_EQ(testOmpTaskSequential.validation(), true);
  testOmpTaskSequential.pre_processing();
  testOmpTaskSequential.run();
  testOmpTaskSequential.post_processing();

  // Create data
  std::vector<double> par_res(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataPar->inputs_count.emplace_back(n.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_res.data()));
  taskDataPar->outputs_count.emplace_back(par_res.size());

  // Create Task
  mortina_a_integral_trapezoid_tbb::TestTBBTaskParallelMortinaIntegralTrapezoid testOmpTaskParallel(taskDataPar,
                                                                                                    x_y_quadro);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  ASSERT_NEAR(par_res[0], out[0], 0.0001);
}

TEST(Parallel_mortina_a_int_trapizoid_TBB, Test_2) {
  double a1 = 0.0;
  double b1 = 3.14159265358979323846;
  double a2 = 0.0;
  double b2 = 3.14159265358979323846;

  int n1 = 100;
  int n2 = 100;

  double res = 2 * 3.14159265358979323846;

  auto sin_cos = [](double x, double y) { return sin(x) + cos(y); };

  // Create data
  std::vector<double> in = {a1, b1, a2, b2};
  std::vector<double> out(1, res);
  std::vector<int> n = {n1, n2};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  mortina_a_integral_trapezoid_tbb::TestTBBTaskSequentialMortinaIntegralTrapezoid testTaskSequential(taskDataSeq,
                                                                                                     sin_cos);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create data
  std::vector<double> par_res(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataPar->inputs_count.emplace_back(n.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_res.data()));
  taskDataPar->outputs_count.emplace_back(par_res.size());

  // Create Task
  mortina_a_integral_trapezoid_tbb::TestTBBTaskParallelMortinaIntegralTrapezoid testOmpTaskParallel(taskDataPar,
                                                                                                    sin_cos);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  ASSERT_NEAR(par_res[0], out[0], 0.0001);
}

TEST(Parallel_mortina_a_int_trapizoid_TBB, Test_3) {
  double a1 = 0.0;
  double b1 = 1.0;
  double a2 = 0.0;
  double b2 = 1.0;

  int n1 = 10;
  int n2 = 10;

  double res = 0.25;

  auto x_mul_y = [](double x, double y) { return x * y; };

  // Create data
  std::vector<double> in = {a1, b1, a2, b2};
  std::vector<double> out(1, res);
  std::vector<int> n = {n1, n2};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  mortina_a_integral_trapezoid_tbb::TestTBBTaskSequentialMortinaIntegralTrapezoid testTaskSequential(taskDataSeq,
                                                                                                     x_mul_y);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create data
  std::vector<double> par_res(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataPar->inputs_count.emplace_back(n.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_res.data()));
  taskDataPar->outputs_count.emplace_back(par_res.size());

  // Create Task
  mortina_a_integral_trapezoid_tbb::TestTBBTaskParallelMortinaIntegralTrapezoid testOmpTaskParallel(taskDataPar,
                                                                                                    x_mul_y);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  ASSERT_NEAR(par_res[0], out[0], 0.0001);
}

TEST(Parallel_mortina_a_int_trapizoid_TBB, Test_4) {
  double a1 = 0.0;
  double b1 = 3.14159265358979323846;
  double a2 = 0.0;
  double b2 = 1;

  int n1 = 50;
  int n2 = 50;

  double res = 2.2685;

  auto x_mult_sinY = [](double x, double y) { return x * sin(y); };
  // func function = x_mult_sinY;

  // Create data
  std::vector<double> in = {a1, b1, a2, b2};
  std::vector<double> out(1, res);
  std::vector<int> n = {n1, n2};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  mortina_a_integral_trapezoid_tbb::TestTBBTaskSequentialMortinaIntegralTrapezoid testTaskSequential(taskDataSeq,
                                                                                                     x_mult_sinY);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create data
  std::vector<double> par_res(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataPar->inputs_count.emplace_back(n.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_res.data()));
  taskDataPar->outputs_count.emplace_back(par_res.size());

  // Create Task
  mortina_a_integral_trapezoid_tbb::TestTBBTaskParallelMortinaIntegralTrapezoid testOmpTaskParallel(taskDataPar,
                                                                                                    x_mult_sinY);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  ASSERT_NEAR(par_res[0], out[0], 0.0001);
}

TEST(Parallel_mortina_a_int_trapizoid_TBB, Test_5) {
  double a1 = 1.0;
  double b1 = 2.0;
  double a2 = 1.0;
  double b2 = 2.0;

  int n1 = 300;
  int n2 = 300;

  double res = 1.039;

  auto x_del_y = [](double x, double y) { return x / y; };

  // Create data
  std::vector<double> in = {a1, b1, a2, b2};
  std::vector<double> out(1, res);
  std::vector<int> n = {n1, n2};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  mortina_a_integral_trapezoid_tbb::TestTBBTaskSequentialMortinaIntegralTrapezoid testTaskSequential(taskDataSeq,
                                                                                                     x_del_y);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  // Create data
  std::vector<double> par_res(1, res);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataPar->inputs_count.emplace_back(n.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_res.data()));
  taskDataPar->outputs_count.emplace_back(par_res.size());

  // Create Task
  mortina_a_integral_trapezoid_tbb::TestTBBTaskParallelMortinaIntegralTrapezoid testOmpTaskParallel(taskDataPar,
                                                                                                    x_del_y);
  ASSERT_EQ(testOmpTaskParallel.validation(), true);
  testOmpTaskParallel.pre_processing();
  testOmpTaskParallel.run();
  testOmpTaskParallel.post_processing();

  ASSERT_NEAR(par_res[0], out[0], 0.0001);
}