// Copyright 2024 Petrov Maksim

#include <gtest/gtest.h>

#include <vector>

#include "omp/petrov_m_radix_sort_double/include/ops_omp.hpp"

TEST(Petrov_M_Radix_Sort_OMP, Test_With_Positive_Numbers) {
  // Create data
  std::vector<double> in{2.5, 3.22, 5.32, 1.11};
  std::vector<double> sorted{1.11, 2.5, 3.22, 5.32};
  std::vector<double> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataPar->outputs_count.emplace_back(out.size());

  // Create Task
  PetrovRadixSortDoubleOMP testTaskOMP(taskDataPar);
  ASSERT_EQ(testTaskOMP.validation(), true);
  testTaskOMP.pre_processing();
  testTaskOMP.run();
  testTaskOMP.post_processing();
  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(sorted[i], out[i]);
  }
}

TEST(Petrov_M_Radix_Sort_OMP, Test_With_Negative_Numbers) {
  // Create data
  std::vector<double> in{-2.5, -3.22, -5.32, -1.11};
  std::vector<double> sorted{-5.32, -3.22, -2.5, -1.11};
  std::vector<double> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataPar->outputs_count.emplace_back(out.size());

  // Create Task
  PetrovRadixSortDoubleOMP testTaskOMP(taskDataPar);
  ASSERT_EQ(testTaskOMP.validation(), true);
  testTaskOMP.pre_processing();
  testTaskOMP.run();
  testTaskOMP.post_processing();
  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(sorted[i], out[i]);
  }
}

TEST(Petrov_M_Radix_Sort_OMP, Test_With_Positive_and_Negative_Numbers) {
  // Create data
  std::vector<double> in{2.5, -3.22, 5.32, -1.11};
  std::vector<double> sorted{-3.22, -1.11, 2.5, 5.32};
  std::vector<double> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataPar->outputs_count.emplace_back(out.size());

  // Create Task
  PetrovRadixSortDoubleOMP testTaskOMP(taskDataPar);
  ASSERT_EQ(testTaskOMP.validation(), true);
  testTaskOMP.pre_processing();
  testTaskOMP.run();
  testTaskOMP.post_processing();
  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(sorted[i], out[i]);
  }
}

TEST(Petrov_M_Radix_Sort_OMP, Test_With_Positive_Numbers2) {
  // Create data
  std::vector<double> in{2.55, 3.223, 5.323, 1.116, 2.56, 3.222, 5.322, 1.115};
  std::vector<double> sorted{1.115, 1.116, 2.55, 2.56, 3.222, 3.223, 5.322, 5.323};
  std::vector<double> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataPar->outputs_count.emplace_back(out.size());

  // Create Task
  PetrovRadixSortDoubleOMP testTaskOMP(taskDataPar);
  ASSERT_EQ(testTaskOMP.validation(), true);
  testTaskOMP.pre_processing();
  testTaskOMP.run();
  testTaskOMP.post_processing();
  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(sorted[i], out[i]);
  }
}

TEST(Petrov_M_Radix_Sort_OMP, Test_With_Negative_Numbers2) {
  // Create data
  std::vector<double> in{-1.1, -2.2, -3.3, -4.4, -5.5, -6.6, -7.7, -8.8, -9.9, -10.0};
  std::vector<double> sorted{-10.0, -9.9, -8.8, -7.7, -6.6, -5.5, -4.4, -3.3, -2.2, -1.1};
  std::vector<double> out(in.size());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataPar->outputs_count.emplace_back(out.size());

  // Create Task
  PetrovRadixSortDoubleOMP testTaskOMP(taskDataPar);
  ASSERT_EQ(testTaskOMP.validation(), true);
  testTaskOMP.pre_processing();
  testTaskOMP.run();
  testTaskOMP.post_processing();
  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(sorted[i], out[i]);
  }
}

TEST(Petrov_M_Radix_Sort_OMP, Test_With_Positive_and_Negative_Numbers3) {
  std::vector<double> in{
      -87.5,  56.7,   22.4,   -3.9,   42.2,   -98.345, -77.4,  65.0,   92.1,   -45.6,  33.8,   -71.9,  -24.2,   80.5,
      49.4,   -88.1,  -19.2,  47.3,   -52.6,  156.23,  120.57, 145.76, 177.05, 101.32, 199.56, 178.91, 125.47,  113.28,
      169.38, 104.55, 191.67, 115.98, 180.62, 198.14,  129.93, 138.60, 175.20, 150.87, 110.42, 14.867, -33.199, 60.2,
      29.1,   -67.5,  99.976, -4.7,   -15.2,  73.6,    -66.8,  23.4,   -59.1,  88.7,   -21.5,  -40.9,  10.6,    -11.8,
      35.5,   -81.6,  5.3,    76.4,   -38.0,  91.232,  28.7,   -53.5,  64.3,   -92.4,  18.6,   82.9,   57.8,    -30.6};
  std::vector<double> sorted{
      -98.345, -92.4,   -88.1,  -87.5,  -81.6,  -77.4,  -71.9,  -67.5,  -66.8,  -59.1,  -53.5,  -52.6,  -45.6,  -40.9,
      -38.0,   -33.199, -30.6,  -24.2,  -21.5,  -19.2,  -15.2,  -11.8,  -4.7,   -3.9,   5.3,    10.6,   14.867, 18.6,
      22.4,    23.4,    28.7,   29.1,   33.8,   35.5,   42.2,   47.3,   49.4,   56.7,   57.8,   60.2,   64.3,   65.0,
      73.6,    76.4,    80.5,   82.9,   88.7,   91.232, 92.1,   99.976, 101.32, 104.55, 110.42, 113.28, 115.98, 120.57,
      125.47,  129.93,  138.60, 145.76, 150.87, 156.23, 169.38, 175.20, 177.05, 178.91, 180.62, 191.67, 198.14, 199.56};
  std::vector<double> out(in.size());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataPar->inputs_count.emplace_back(in.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataPar->outputs_count.emplace_back(out.size());

  // Create Task
  PetrovRadixSortDoubleOMP testTaskOMP(taskDataPar);
  ASSERT_EQ(testTaskOMP.validation(), true);
  testTaskOMP.pre_processing();
  testTaskOMP.run();
  testTaskOMP.post_processing();
  for (size_t i = 0; i < out.size(); i++) {
    ASSERT_EQ(sorted[i], out[i]);
  }
}
