// Copyright 2024 Shubin Mikhail

#include "seq/shubin_m_double_crs_mult/include/sparsemat_crs.hpp"

#include <cmath>

SparseMat_CRS::SparseMat_CRS(size_t _row_c, size_t _col_c) {
  row_c = _row_c;
  col_c = _col_c;
}

SparseMat_CRS::SparseMat_CRS(size_t _row_c, size_t _col_c, size_t _nz_c) {
  row_c = _row_c;
  col_c = _col_c;
  nz_c = _nz_c;
}

SparseMat_CRS::SparseMat_CRS(const std::vector<double>& matrix, size_t _row_c, size_t _col_c) {
  row_c = _row_c;
  col_c = _col_c;
  nz_c = 0;
  for (size_t i = 0; i < row_c; i++) {
    row_ind.push_back(val.size());
    for (size_t j = 0; j < col_c; j++) {
      double temp = matrix[i * col_c + j];
      if (std::abs(temp) > PRECISION) {
        val.push_back(temp);
        col_ind.push_back(j);
        nz_c++;
      }
    }
  }
  row_ind.push_back(val.size());
}

SparseMat_CRS random_CRS_mat(size_t _row_c, size_t _col_c, double dens, double _min, double _max) {
  auto nz_max = static_cast<size_t>(static_cast<double>(_row_c * _col_c) * dens);
  std::vector<double> matrix(_row_c * _col_c, 0.0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> value_dist(_min, _max);
  std::uniform_int_distribution<size_t> index_dist(0, _row_c * _col_c - 1);

  for (size_t i = 0; i < nz_max; i++) {
    matrix[index_dist(gen)] = value_dist(gen);
  }

  SparseMat_CRS res(matrix, _row_c, _col_c);

  return res;
}

SparseMat_CRS ident_CRS_mat(size_t _size) {
  SparseMat_CRS res(_size, _size, _size);

  for (size_t i = 0; i < _size; i++) {
    res.val.push_back(1.0);
    res.row_ind.push_back(i);
    res.col_ind.push_back(i);
  }
  res.row_ind.push_back(_size);

  return res;
}

std::vector<double> CRS_to_vector(const SparseMat_CRS& p) {
  std::vector<double> res(p.row_c * p.col_c, 0.0);

  for (size_t i = 0; i < p.row_c; i++) {
    for (size_t j = p.row_ind[i]; j < p.row_ind[i + 1]; j++) {
      res[i * p.col_c + p.col_ind[j]] = p.val[j];
    }
  }

  return res;
}
