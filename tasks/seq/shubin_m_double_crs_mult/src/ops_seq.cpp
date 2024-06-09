// Copyright 2024 Shubin Mikhail

#include "seq/shubin_m_double_crs_mult/include/ops_seq.hpp"

bool SparseMat_CRS_Mult_Seq::validation() {
  internal_order_test();

  size_t mat_in_1_col_c = reinterpret_cast<SparseMat_CRS*>(taskData->inputs[0])->col_c;
  size_t mat_in_2_row_c = reinterpret_cast<SparseMat_CRS*>(taskData->inputs[1])->row_c;

  return (mat_in_1_col_c == mat_in_2_row_c);
}

bool SparseMat_CRS_Mult_Seq::pre_processing() {
  internal_order_test();

  mat_in_1 = reinterpret_cast<SparseMat_CRS*>(taskData->inputs[0]);
  mat_in_2 = reinterpret_cast<SparseMat_CRS*>(taskData->inputs[1]);
  mat_out = reinterpret_cast<SparseMat_CRS*>(taskData->outputs[0]);

  return true;
}

bool SparseMat_CRS_Mult_Seq::run() {
  internal_order_test();

  std::vector<double> curr_row(mat_out->col_c, 0.0);
  for (size_t i = 0; i < mat_in_1->row_c; i++) {
    for (size_t j = mat_in_1->row_ind[i]; j < mat_in_1->row_ind[i + 1]; j++) {
      size_t col = mat_in_1->col_ind[j];
      double temp = mat_in_1->val[j];
      for (size_t k = mat_in_2->row_ind[col]; k < mat_in_2->row_ind[col + 1]; k++) {
        curr_row[mat_in_2->col_ind[k]] += temp * mat_in_2->val[k];
      }
    }

    mat_out->row_ind.push_back(mat_out->val.size());
    for (size_t j = 0; j < curr_row.size(); j++) {
      if (std::abs(curr_row[j]) > PRECISION) {
        mat_out->val.push_back(curr_row[j]);
        mat_out->col_ind.push_back(j);
        curr_row[j] = 0.0;
      }
    }
  }
  mat_out->row_ind.push_back(mat_out->val.size());

  return true;
}

bool SparseMat_CRS_Mult_Seq::post_processing() {
  internal_order_test();

  return true;
}
