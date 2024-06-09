// Copyright 2024 Veselov Mikhail
#include "omp/veselov_m_matrcomplexmultyCCS_omp/include/ops_omp.hpp"

#include <thread>

using namespace VeselovOmp;

bool SparseMatrixComplexMultiOMPSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[1] == taskData->inputs_count[2] &&
         taskData->outputs_count[0] == taskData->inputs_count[0] &&
         taskData->outputs_count[1] == taskData->inputs_count[3];
}

bool SparseMatrixComplexMultiOMPSequential::pre_processing() {
  internal_order_test();

  auto* mat1 = reinterpret_cast<Complex*>(taskData->inputs[0]);
  auto* mat2 = reinterpret_cast<Complex*>(taskData->inputs[1]);

  numRows1 = taskData->inputs_count[0];
  numCols1 = taskData->inputs_count[1];
  numRows2 = taskData->inputs_count[2];
  numCols2 = taskData->inputs_count[3];

  res = new Complex[numRows1 * numCols2]{{0.0, 0.0}};

  int notNullNumbers = 0;
  for (int j = 0; j < numCols1; ++j) {
    cols1.push_back(notNullNumbers);
    for (int i = 0; i < numRows1; ++i) {
      int index = i * numRows1 + j;
      if (mat1[index].real != 0 || mat1[index].imag != 0) {
        val1.push_back(mat1[index]);
        rows1.push_back(i);
        notNullNumbers++;
      }
    }
  }
  cols1.push_back(notNullNumbers);

  notNullNumbers = 0;
  for (int j = 0; j < numCols2; ++j) {
    cols2.push_back(notNullNumbers);
    for (int i = 0; i < numRows2; ++i) {
      int index = i * numRows2 + j;
      if (mat2[index].real != 0 || mat2[index].imag != 0) {
        val2.push_back(mat2[index]);
        rows2.push_back(i);
        notNullNumbers++;
      }
    }
  }
  cols2.push_back(notNullNumbers);

  return true;
}

bool SparseMatrixComplexMultiOMPSequential::run() {
  internal_order_test();

  for (int j = 0; j < numCols2; j++) {
    for (int k = cols2[j]; k < cols2[j + 1]; k++) {
      Complex b = val2[k];
      int row_b = rows2[k];
      for (int i = cols1[row_b]; i < cols1[row_b + 1]; i++) {
        Complex a = val1[i];
        int row_a = rows1[i];
        res[row_a * numCols2 + j].real += a.real * b.real - a.imag * b.imag;
        res[row_a * numCols2 + j].imag += a.real * b.imag + a.imag * b.real;
      }
    }
  }

  return true;
}

bool SparseMatrixComplexMultiOMPSequential::post_processing() {
  internal_order_test();

  auto* out_ptr = reinterpret_cast<Complex*>(taskData->outputs[0]);
  for (int i = 0; i < numRows1 * numCols2; i++) {
    out_ptr[i] = res[i];
  }

  delete[] res;

  return true;
}

bool SparseMatrixComplexMultiOMPParallel::validation() {
  internal_order_test();
  return taskData->inputs_count[1] == taskData->inputs_count[2] &&
         taskData->outputs_count[0] == taskData->inputs_count[0] &&
         taskData->outputs_count[1] == taskData->inputs_count[3];
}

bool SparseMatrixComplexMultiOMPParallel::pre_processing() {
  internal_order_test();

  auto* mat1 = reinterpret_cast<Complex*>(taskData->inputs[0]);
  auto* mat2 = reinterpret_cast<Complex*>(taskData->inputs[1]);

  numRows1 = taskData->inputs_count[0];
  numCols1 = taskData->inputs_count[1];
  numRows2 = taskData->inputs_count[2];
  numCols2 = taskData->inputs_count[3];

  res = new Complex[numRows1 * numCols2]{{0.0, 0.0}};

  int notNullNumbers = 0;
  for (int j = 0; j < numCols1; ++j) {
    cols1.push_back(notNullNumbers);
    for (int i = 0; i < numRows1; ++i) {
      int index = i * numRows1 + j;
      if (mat1[index].real != 0 || mat1[index].imag != 0) {
        val1.push_back(mat1[index]);
        rows1.push_back(i);
        notNullNumbers++;
      }
    }
  }
  cols1.push_back(notNullNumbers);

  notNullNumbers = 0;
  for (int j = 0; j < numCols2; ++j) {
    cols2.push_back(notNullNumbers);
    for (int i = 0; i < numRows2; ++i) {
      int index = i * numRows2 + j;
      if (mat2[index].real != 0 || mat2[index].imag != 0) {
        val2.push_back(mat2[index]);
        rows2.push_back(i);
        notNullNumbers++;
      }
    }
  }
  cols2.push_back(notNullNumbers);

  return true;
}

bool SparseMatrixComplexMultiOMPParallel::run() {
  internal_order_test();

#pragma omp parallel for
  for (int j = 0; j < numCols2; j++) {
    for (int k = cols2[j]; k < cols2[j + 1]; k++) {
      Complex b = val2[k];
      int row_b = rows2[k];
      for (int i = cols1[row_b]; i < cols1[row_b + 1]; i++) {
        Complex a = val1[i];
        int row_a = rows1[i];
        res[row_a * numCols2 + j].real += a.real * b.real - a.imag * b.imag;
        res[row_a * numCols2 + j].imag += a.real * b.imag + a.imag * b.real;
      }
    }
  }

  return true;
}

bool SparseMatrixComplexMultiOMPParallel::post_processing() {
  internal_order_test();

  auto* out_ptr = reinterpret_cast<Complex*>(taskData->outputs[0]);
  for (int i = 0; i < numRows1 * numCols2; i++) {
    out_ptr[i] = res[i];
  }

  delete[] res;

  return true;
}
