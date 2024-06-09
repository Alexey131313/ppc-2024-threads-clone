// Copyright 2024 Ionova Ekatetina
#include "omp/ionova_e_sparse_matr_multi_crs_complex_omp/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool SparseMatrixComplexMultiSequentialOmp::validation() {
  internal_order_test();
  return taskData->inputs_count[1] == taskData->inputs_count[2] &&
         taskData->outputs_count[0] == taskData->inputs_count[0] &&
         taskData->outputs_count[1] == taskData->inputs_count[3];
}

bool SparseMatrixComplexMultiSequentialOmp::pre_processing() {
  internal_order_test();

  auto* matrix1 = reinterpret_cast<Complex*>(taskData->inputs[0]);
  auto* matrix2 = reinterpret_cast<Complex*>(taskData->inputs[1]);
  numRows1 = taskData->inputs_count[0];
  numCols1 = taskData->inputs_count[1];
  numRows2 = taskData->inputs_count[2];
  numCols2 = taskData->inputs_count[3];

  result = new Complex[numRows1 * numCols2]{{0.0, 0.0}};

  int notNullNumbers = 0;
  for (int i = 0; i < numRows1; ++i) {
    rowPtr1.push_back(notNullNumbers);
    for (int j = 0; j < numCols1; ++j) {
      int index = i * numCols1 + j;
      if (matrix1[index].real != 0 || matrix1[index].imag != 0) {
        values1.push_back(matrix1[index]);
        colPtr1.push_back(j);
        notNullNumbers++;
      }
    }
  }
  rowPtr1.push_back(notNullNumbers);

  notNullNumbers = 0;
  for (int i = 0; i < numRows2; ++i) {
    rowPtr2.push_back(notNullNumbers);
    for (int j = 0; j < numCols2; ++j) {
      int index = i * numCols2 + j;
      if (matrix2[index].real != 0 || matrix2[index].imag != 0) {
        values2.push_back(matrix2[index]);
        colPtr2.push_back(j);
        notNullNumbers++;
      }
    }
  }
  rowPtr2.push_back(notNullNumbers);

  return true;
}

bool SparseMatrixComplexMultiSequentialOmp::run() {
  internal_order_test();

  values3.clear();
  rowPtr3.clear();
  colPtr3.clear();
  for (int i = 0; i < numRows1; i++) {
    for (int j = rowPtr1[i]; j < rowPtr1[i + 1]; j++) {
      int row1 = i;
      int col1 = colPtr1[j];
      Complex val1 = values1[j];
      for (int k = rowPtr2[col1]; k < rowPtr2[col1 + 1]; k++) {
        int col2 = colPtr2[k];
        Complex val2 = values2[k];
        int index = row1 * numCols2 + col2;
        result[index].real += val1.real * val2.real - val1.imag * val2.imag;
        result[index].imag += val1.imag * val2.real + val1.real * val2.imag;
      }
    }
  }

  for (int i = 0; i < numRows1; i++) {
    rowPtr3.push_back(values3.size());
    for (int j = 0; j < numCols2; j++) {
      int ind = i * numCols2 + j;
      if (result[ind].real != 0.0 || result[ind].imag != 0.0) {
        values3.push_back(result[ind]);
        colPtr3.push_back(j);
      }
    }
  }
  rowPtr3.push_back(values3.size());

  return true;
}

bool SparseMatrixComplexMultiSequentialOmp::post_processing() {
  internal_order_test();

  auto* out_ptr = reinterpret_cast<Complex*>(taskData->outputs[0]);
  for (int i = 0; i < numRows1 * numCols2; i++) {
    out_ptr[i] = result[i];
  }

  delete[] result;

  return true;
}

bool SparseMatrixComplexMultiParallelOmp::validation() {
  internal_order_test();
  return taskData->inputs_count[1] == taskData->inputs_count[2] &&
         taskData->outputs_count[0] == taskData->inputs_count[0] &&
         taskData->outputs_count[1] == taskData->inputs_count[3];
}

bool SparseMatrixComplexMultiParallelOmp::pre_processing() {
  internal_order_test();

  auto* matrix1 = reinterpret_cast<Complex*>(taskData->inputs[0]);
  auto* matrix2 = reinterpret_cast<Complex*>(taskData->inputs[1]);
  numRows1 = taskData->inputs_count[0];
  numCols1 = taskData->inputs_count[1];
  numRows2 = taskData->inputs_count[2];
  numCols2 = taskData->inputs_count[3];

  result = new Complex[numRows1 * numCols2]{{0.0, 0.0}};

  int notNullNumbers = 0;
  for (int i = 0; i < numRows1; ++i) {
    rowPtr1.push_back(notNullNumbers);
    for (int j = 0; j < numCols1; ++j) {
      int index = i * numCols1 + j;
      if (matrix1[index].real != 0 || matrix1[index].imag != 0) {
        values1.push_back(matrix1[index]);
        colPtr1.push_back(j);
        notNullNumbers++;
      }
    }
  }
  rowPtr1.push_back(notNullNumbers);

  notNullNumbers = 0;
  for (int i = 0; i < numRows2; ++i) {
    rowPtr2.push_back(notNullNumbers);
    for (int j = 0; j < numCols2; ++j) {
      int index = i * numCols2 + j;
      if (matrix2[index].real != 0 || matrix2[index].imag != 0) {
        values2.push_back(matrix2[index]);
        colPtr2.push_back(j);
        notNullNumbers++;
      }
    }
  }
  rowPtr2.push_back(notNullNumbers);

  return true;
}

bool SparseMatrixComplexMultiParallelOmp::run() {
  internal_order_test();

  values3.clear();
  rowPtr3.clear();
  colPtr3.clear();
#pragma omp parallel for
  for (int i = 0; i < numRows1; i++) {
    for (int j = rowPtr1[i]; j < rowPtr1[i + 1]; j++) {
      int row1 = i;
      int col1 = colPtr1[j];
      Complex val1 = values1[j];
      for (int k = rowPtr2[col1]; k < rowPtr2[col1 + 1]; k++) {
        int col2 = colPtr2[k];
        Complex val2 = values2[k];
        int index = row1 * numCols2 + col2;
        result[index].real += val1.real * val2.real - val1.imag * val2.imag;
        result[index].imag += val1.imag * val2.real + val1.real * val2.imag;
      }
    }
  }

  for (int i = 0; i < numRows1; i++) {
    rowPtr3.push_back(values3.size());
    for (int j = 0; j < numCols2; j++) {
      int ind = i * numCols2 + j;
      if (result[ind].real != 0.0 || result[ind].imag != 0.0) {
        values3.push_back(result[ind]);
        colPtr3.push_back(j);
      }
    }
  }
  rowPtr3.push_back(values3.size());

  return true;
}

bool SparseMatrixComplexMultiParallelOmp::post_processing() {
  internal_order_test();

  auto* out_ptr = reinterpret_cast<Complex*>(taskData->outputs[0]);
  for (int i = 0; i < numRows1 * numCols2; i++) {
    out_ptr[i] = result[i];
  }

  delete[] result;

  return true;
}
