// Copyright 2024 Mirzakhmedov Alexander
#include <complex>
#include <vector>

#include "core/task/include/task.hpp"

struct MatrixInCCS_Sparse {
  std::vector<std::complex<double>> val{};
  std::vector<int> row;
  std::vector<int> col_p;

  int r_n;
  int c_n;
  int n_z;

  MatrixInCCS_Sparse(int _r_n = 0, int _c_n = 0, int _n_z = 0) {
    c_n = _c_n;
    r_n = _r_n;
    n_z = _n_z;
    row.resize(n_z);
    col_p.resize(r_n + 1);
    val.resize(n_z);
  }
};

class Seq_MatMultCCS : public ppc::core::Task {
 public:
  explicit Seq_MatMultCCS(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  MatrixInCCS_Sparse *M1, *M2, *M3;
};
