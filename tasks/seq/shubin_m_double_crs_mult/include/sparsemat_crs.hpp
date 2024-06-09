// Copyright 2024 Shubin Mikhail

#ifndef TASKS_SEQ_SHUBIN_M_DOUBLE_CRS_MULT_INCLUDE_SPARSEMAT_CRS_HPP_
#define TASKS_SEQ_SHUBIN_M_DOUBLE_CRS_MULT_INCLUDE_SPARSEMAT_CRS_HPP_

#include <random>
#include <vector>

#define PRECISION 1e-8

class SparseMat_CRS {
 public:
  std::vector<double> val;
  std::vector<size_t> col_ind;
  std::vector<size_t> row_ind;
  size_t row_c;
  size_t col_c;
  size_t nz_c;

  SparseMat_CRS(size_t _row_c, size_t _col_c);
  SparseMat_CRS(size_t _row_c, size_t _col_c, size_t _nz_c);
  SparseMat_CRS(const std::vector<double>& matrix, size_t _row_c, size_t _col_c);
  ~SparseMat_CRS() = default;
};

SparseMat_CRS random_CRS_mat(size_t _row_c, size_t _col_c, double dens = 0.1, double _min = -100.0,
                             double _max = 100.0);

SparseMat_CRS ident_CRS_mat(size_t _size);

std::vector<double> CRS_to_vector(const SparseMat_CRS& p);

#endif  // TASKS_SEQ_SHUBIN_M_DOUBLE_CRS_MULT_INCLUDE_SPARSEMAT_CRS_HPP_
