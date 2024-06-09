// Copyright 2024 Isaev Dmitriy
#include <gtest/gtest.h>

#include <vector>

#include "omp/isaev_d_sparse_multe_double_crs/include/ops_omp.hpp"

TEST(isaev_d_sparse_multe_double_crs_omp, Test1) {
  // Create data
  IsaevOMP::SparseMatrix a;
  a.rows = 6;
  a.columns = 6;
  a.row_pointers = {0, 2, 4, 4, 6, 7, 8};
  a.column_indices = {1, 3, 0, 4, 0, 2, 3, 5};
  a.values = {8.0, 2.4, 7.7, 1.3, 1.5, 4.1, 2.5, 6.4};

  IsaevOMP::SparseMatrix b;
  b.rows = 6;
  b.columns = 6;
  b.row_pointers = {0, 1, 2, 4, 6, 8, 9};
  b.column_indices = {3, 1, 0, 4, 1, 4, 0, 5, 0};
  b.values = {8.5, 4.7, 6.0, 5.1, 2.3, 1.8, 1.1, 0.5, 3.2};
  IsaevOMP::SparseMatrix c_seq;

  IsaevOMP::SparseMatrix res;
  res.rows = 6;
  res.columns = 6;
  res.row_pointers = {0, 2, 5, 5, 8, 10, 11};
  res.column_indices = {1, 4, 0, 3, 5, 0, 3, 4, 1, 4, 0};
  res.values = {43.12, 4.32, 1.43, 65.45, 0.65, 24.6, 12.75, 20.91, 5.75, 4.5, 20.48};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataSeq->inputs_count.emplace_back(a.rows);
  taskDataSeq->inputs_count.emplace_back(a.columns);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataSeq->inputs_count.emplace_back(b.rows);
  taskDataSeq->inputs_count.emplace_back(b.columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c_seq));
  taskDataSeq->outputs_count.emplace_back(res.values.size());

  // Create Task
  IsaevOMP::SparseMultDoubleCRSompSeq ompTaskSequential(taskDataSeq);
  ASSERT_EQ(ompTaskSequential.validation(), true);
  ompTaskSequential.pre_processing();
  ompTaskSequential.run();
  ompTaskSequential.post_processing();

  // Create data
  IsaevOMP::SparseMatrix c_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataPar->inputs_count.emplace_back(a.rows);
  taskDataPar->inputs_count.emplace_back(a.columns);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataPar->inputs_count.emplace_back(b.rows);
  taskDataPar->inputs_count.emplace_back(b.columns);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c_par));
  taskDataPar->outputs_count.emplace_back(res.values.size());

  // Create Task
  IsaevOMP::SparseMultDoubleCRSompParallel ompTaskParallel(taskDataPar);
  ASSERT_EQ(ompTaskParallel.validation(), true);
  ompTaskParallel.pre_processing();
  ompTaskParallel.run();
  ompTaskParallel.post_processing();

  for (size_t i = 0; i < c_seq.values.size(); i++) {
    EXPECT_DOUBLE_EQ(c_seq.values[i], c_par.values[i]);
  }
}

TEST(isaev_d_sparse_multe_double_crs_omp, Test2) {
  IsaevOMP::SparseMatrix a;
  a.rows = 3;
  a.columns = 4;
  a.row_pointers = {0, 1, 2, 3};
  a.column_indices = {0, 2, 3};
  a.values = {7.2, 10.5, 13.1};

  IsaevOMP::SparseMatrix b;
  b.rows = 4;
  b.columns = 3;
  b.row_pointers = {0, 1, 1, 2, 3};
  b.column_indices = {2, 0, 1};
  b.values = {8.3, 5.7, 12.4};
  IsaevOMP::SparseMatrix c_seq;

  IsaevOMP::SparseMatrix res;
  res.rows = 3;
  res.columns = 3;
  res.row_pointers = {0, 1, 2, 3};
  res.column_indices = {2, 0, 1};
  res.values = {59.76, 59.85, 162.44};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataSeq->inputs_count.emplace_back(a.rows);
  taskDataSeq->inputs_count.emplace_back(a.columns);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataSeq->inputs_count.emplace_back(b.rows);
  taskDataSeq->inputs_count.emplace_back(b.columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c_seq));
  taskDataSeq->outputs_count.emplace_back(res.values.size());

  // Create Task
  IsaevOMP::SparseMultDoubleCRSompSeq ompTaskSequential(taskDataSeq);
  ASSERT_EQ(ompTaskSequential.validation(), true);
  ompTaskSequential.pre_processing();
  ompTaskSequential.run();
  ompTaskSequential.post_processing();

  // Create data
  IsaevOMP::SparseMatrix c_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataPar->inputs_count.emplace_back(a.rows);
  taskDataPar->inputs_count.emplace_back(a.columns);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataPar->inputs_count.emplace_back(b.rows);
  taskDataPar->inputs_count.emplace_back(b.columns);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c_par));
  taskDataPar->outputs_count.emplace_back(res.values.size());

  // Create Task
  IsaevOMP::SparseMultDoubleCRSompParallel ompTaskParallel(taskDataPar);
  ASSERT_EQ(ompTaskParallel.validation(), true);
  ompTaskParallel.pre_processing();
  ompTaskParallel.run();
  ompTaskParallel.post_processing();

  for (size_t i = 0; i < c_seq.values.size(); i++) {
    EXPECT_DOUBLE_EQ(c_seq.values[i], c_par.values[i]);
  }
}

TEST(isaev_d_sparse_multe_double_crs_omp, Test3) {
  IsaevOMP::SparseMatrix a;
  a.rows = 4;
  a.columns = 5;
  a.row_pointers = {0, 2, 4, 5, 7};
  a.column_indices = {0, 3, 2, 4, 1, 0, 3};
  a.values = {3.9, 1.2, 10.45, 5.1, -2.3, 8.6, -0.7};

  IsaevOMP::SparseMatrix b;
  b.rows = 5;
  b.columns = 4;
  b.row_pointers = {0, 1, 2, 3, 5, 6};
  b.column_indices = {2, 1, 2, 0, 3, 1};
  b.values = {-1.1, 0.5, 3.1, 0.8, -2.7, 1.4};
  IsaevOMP::SparseMatrix c_seq;

  IsaevOMP::SparseMatrix res;
  res.rows = 4;
  res.columns = 4;
  res.row_pointers = {0, 3, 5, 6, 9};
  res.column_indices = {0, 2, 3, 1, 2, 1, 0, 2, 3};
  res.values = {0.96, -4.29, -3.24, 7.14, 32.395, -1.15, -0.56, -9.46, 1.89};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataSeq->inputs_count.emplace_back(a.rows);
  taskDataSeq->inputs_count.emplace_back(a.columns);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataSeq->inputs_count.emplace_back(b.rows);
  taskDataSeq->inputs_count.emplace_back(b.columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c_seq));
  taskDataSeq->outputs_count.emplace_back(res.values.size());

  // Create Task
  IsaevOMP::SparseMultDoubleCRSompSeq ompTaskSequential(taskDataSeq);
  ASSERT_EQ(ompTaskSequential.validation(), true);
  ompTaskSequential.pre_processing();
  ompTaskSequential.run();
  ompTaskSequential.post_processing();

  // Create data
  IsaevOMP::SparseMatrix c_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataPar->inputs_count.emplace_back(a.rows);
  taskDataPar->inputs_count.emplace_back(a.columns);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataPar->inputs_count.emplace_back(b.rows);
  taskDataPar->inputs_count.emplace_back(b.columns);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c_par));
  taskDataPar->outputs_count.emplace_back(res.values.size());

  // Create Task
  IsaevOMP::SparseMultDoubleCRSompParallel ompTaskParallel(taskDataPar);
  ASSERT_EQ(ompTaskParallel.validation(), true);
  ompTaskParallel.pre_processing();
  ompTaskParallel.run();
  ompTaskParallel.post_processing();

  for (size_t i = 0; i < c_seq.values.size(); i++) {
    EXPECT_DOUBLE_EQ(c_seq.values[i], c_par.values[i]);
  }
}

TEST(isaev_d_sparse_multe_double_crs_omp, Test4) {
  IsaevOMP::SparseMatrix a;
  a.rows = 5;
  a.columns = 5;
  a.row_pointers = {0, 2, 3, 3, 4, 5};
  a.column_indices = {0, 2, 1, 2, 1};
  a.values = {2.5, -1.7, 3.4, 0.8, 0.6};

  IsaevOMP::SparseMatrix b;
  b.rows = 5;
  b.columns = 5;
  b.row_pointers = {0, 1, 2, 3, 4, 5};
  b.column_indices = {0, 1, 2, 3, 4};
  b.values = {1.0, 1.0, 1.0, 1.0, 1.0};
  IsaevOMP::SparseMatrix c_seq;

  IsaevOMP::SparseMatrix res;
  res.rows = 5;
  res.columns = 5;
  res.row_pointers = {0, 2, 3, 3, 4, 5};
  res.column_indices = {0, 2, 1, 2, 1};
  res.values = {2.5, -1.7, 3.4, 0.8, 0.6};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataSeq->inputs_count.emplace_back(a.rows);
  taskDataSeq->inputs_count.emplace_back(a.columns);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataSeq->inputs_count.emplace_back(b.rows);
  taskDataSeq->inputs_count.emplace_back(b.columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c_seq));
  taskDataSeq->outputs_count.emplace_back(res.values.size());

  // Create Task
  IsaevOMP::SparseMultDoubleCRSompSeq ompTaskSequential(taskDataSeq);
  ASSERT_EQ(ompTaskSequential.validation(), true);
  ompTaskSequential.pre_processing();
  ompTaskSequential.run();
  ompTaskSequential.post_processing();

  // Create data
  IsaevOMP::SparseMatrix c_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataPar->inputs_count.emplace_back(a.rows);
  taskDataPar->inputs_count.emplace_back(a.columns);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataPar->inputs_count.emplace_back(b.rows);
  taskDataPar->inputs_count.emplace_back(b.columns);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c_par));
  taskDataPar->outputs_count.emplace_back(res.values.size());

  // Create Task
  IsaevOMP::SparseMultDoubleCRSompParallel ompTaskParallel(taskDataPar);
  ASSERT_EQ(ompTaskParallel.validation(), true);
  ompTaskParallel.pre_processing();
  ompTaskParallel.run();
  ompTaskParallel.post_processing();

  for (size_t i = 0; i < c_seq.values.size(); i++) {
    EXPECT_DOUBLE_EQ(c_seq.values[i], c_par.values[i]);
  }
}

TEST(isaev_d_sparse_multe_double_crs_omp, Test5) {
  IsaevOMP::SparseMatrix a;
  a.rows = 3;
  a.columns = 3;
  a.row_pointers = {0, 1, 2, 3};
  a.column_indices = {0, 1, 2};
  a.values = {1.0, 2.0, 3.0};

  IsaevOMP::SparseMatrix b;
  b.rows = 3;
  b.columns = 3;
  b.row_pointers = {0, 1, 2, 3};
  b.column_indices = {0, 1, 2};
  b.values = {2.0, 2.0, 2.0};
  IsaevOMP::SparseMatrix c_seq;

  IsaevOMP::SparseMatrix res;
  res.rows = 3;
  res.columns = 3;
  res.row_pointers = {0, 1, 2, 3};
  res.column_indices = {0, 1, 2};
  res.values = {2.0, 4.0, 6.0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataSeq->inputs_count.emplace_back(a.rows);
  taskDataSeq->inputs_count.emplace_back(a.columns);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataSeq->inputs_count.emplace_back(b.rows);
  taskDataSeq->inputs_count.emplace_back(b.columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c_seq));
  taskDataSeq->outputs_count.emplace_back(res.values.size());

  // Create Task
  IsaevOMP::SparseMultDoubleCRSompSeq ompTaskSequential(taskDataSeq);
  ASSERT_EQ(ompTaskSequential.validation(), true);
  ompTaskSequential.pre_processing();
  ompTaskSequential.run();
  ompTaskSequential.post_processing();

  // Create data
  IsaevOMP::SparseMatrix c_par;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataPar->inputs_count.emplace_back(a.rows);
  taskDataPar->inputs_count.emplace_back(a.columns);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataPar->inputs_count.emplace_back(b.rows);
  taskDataPar->inputs_count.emplace_back(b.columns);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c_par));
  taskDataPar->outputs_count.emplace_back(res.values.size());

  // Create Task
  IsaevOMP::SparseMultDoubleCRSompParallel ompTaskParallel(taskDataPar);
  ASSERT_EQ(ompTaskParallel.validation(), true);
  ompTaskParallel.pre_processing();
  ompTaskParallel.run();
  ompTaskParallel.post_processing();

  for (size_t i = 0; i < c_seq.values.size(); i++) {
    EXPECT_DOUBLE_EQ(c_seq.values[i], c_par.values[i]);
  }
}
