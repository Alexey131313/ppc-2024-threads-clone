// Copyright 2024 Isaev Dmitriy
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace IsaevSeq {
struct SparseMatrix {
  std::vector<double> values{};
  std::vector<int> column_indices{};
  std::vector<int> row_pointers{};

  int rows = 0;
  int columns = 0;
};

class SparseMultDoubleCRS : public ppc::core::Task {
 public:
  explicit SparseMultDoubleCRS(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  SparseMatrix *A{}, *B{}, *C{};
};

SparseMatrix getRandomMatrix(int _rows, int _columns, double chance, int seed);

}  // namespace IsaevSeq