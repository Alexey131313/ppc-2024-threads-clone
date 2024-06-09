// Copyright 2024 Simonyan Suren

#include "tbb/simonyan_s_sparse_matr_multi_ccs_tbb/include/ccs_mat_multy.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>

using namespace std::chrono_literals;

using namespace std;

bool SparseTBBMatrixMultiSequential::pre_processing() {
  internal_order_test();
  auto* matrix1 = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* matrix2 = reinterpret_cast<double*>(taskData->inputs[1]);
  numRows1 = taskData->inputs_count[0];
  numCols1 = taskData->inputs_count[1];
  numRows2 = taskData->inputs_count[2];
  numCols2 = taskData->inputs_count[3];
  numRows3 = numRows1;
  numCols3 = numCols2;

  result = new double[numRows1 * numCols2]{0};

  int notNullNumbers = 0;
  for (int j = 0; j < numCols1; ++j) {
    colPtr1.push_back(notNullNumbers);
    for (int i = 0; i < numRows1; ++i) {
      int index = i * numRows1 + j;
      if (matrix1[index] != 0) {
        values1.push_back(matrix1[index]);
        rows1.push_back(i);
        notNullNumbers++;
      }
    }
  }
  colPtr1.push_back(notNullNumbers);

  notNullNumbers = 0;
  for (int j = 0; j < numCols2; ++j) {
    colPtr2.push_back(notNullNumbers);
    for (int i = 0; i < numRows2; ++i) {
      int index = i * numRows2 + j;
      if (matrix2[index] != 0) {
        values2.push_back(matrix2[index]);
        rows2.push_back(i);
        notNullNumbers++;
      }
    }
  }
  colPtr2.push_back(notNullNumbers);

  return true;
}

bool SparseTBBMatrixMultiSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[1] == taskData->inputs_count[2] &&
         taskData->outputs_count[0] == taskData->inputs_count[0] &&
         taskData->outputs_count[1] == taskData->inputs_count[3];
}

bool SparseTBBMatrixMultiSequential::run() {
  internal_order_test();

  values3.clear();
  rows3.clear();
  colPtr3.clear();

  for (int j = 0; j < numCols1; j++) {
    for (int k = colPtr2[j]; k < colPtr2[j + 1]; k++) {
      int column2 = j;
      int row2 = rows2[k];
      for (int l = colPtr1[row2]; l < colPtr1[row2 + 1]; l++) {
        int row1 = rows1[l];
        double val1 = values1[l];
        double val2 = values2[k];
        int index = row1 * numCols2 + column2;
        result[index] += val1 * val2;
      }
    }
  }

  for (int j = 0; j < numCols2; j++) {
    colPtr3.push_back(values3.size());
    for (int i = 0; i < numRows1; i++) {
      int ind = i * numCols2 + j;
      if (result[ind] != 0.0) {
        values3.push_back(result[ind]);
        rows3.push_back(i);
      }
    }
  }
  colPtr3.push_back(values3.size());

  return true;
}

bool SparseTBBMatrixMultiSequential::post_processing() {
  internal_order_test();

  auto* out_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  for (int i = 0; i < numRows3 * numCols3; i++) {
    out_ptr[i] = result[i];
  }

  delete[] result;

  return true;
}

bool SparseTBBMatrixMultiParallel::pre_processing() {
  internal_order_test();
  auto* matrix1 = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* matrix2 = reinterpret_cast<double*>(taskData->inputs[1]);
  numRows1 = taskData->inputs_count[0];
  numCols1 = taskData->inputs_count[1];
  numRows2 = taskData->inputs_count[2];
  numCols2 = taskData->inputs_count[3];
  numRows3 = numRows1;
  numCols3 = numCols2;

  result = new double[numRows1 * numCols2]{0};

  int notNullNumbers = 0;
  for (int j = 0; j < numCols1; ++j) {
    colPtr1.push_back(notNullNumbers);
    for (int i = 0; i < numRows1; ++i) {
      int index = i * numRows1 + j;
      if (matrix1[index] != 0) {
        values1.push_back(matrix1[index]);
        rows1.push_back(i);
        notNullNumbers++;
      }
    }
  }
  colPtr1.push_back(notNullNumbers);

  notNullNumbers = 0;
  for (int j = 0; j < numCols2; ++j) {
    colPtr2.push_back(notNullNumbers);
    for (int i = 0; i < numRows2; ++i) {
      int index = i * numRows2 + j;
      if (matrix2[index] != 0) {
        values2.push_back(matrix2[index]);
        rows2.push_back(i);
        notNullNumbers++;
      }
    }
  }
  colPtr2.push_back(notNullNumbers);

  return true;
}

bool SparseTBBMatrixMultiParallel::validation() {
  internal_order_test();
  return taskData->inputs_count[1] == taskData->inputs_count[2] &&
         taskData->outputs_count[0] == taskData->inputs_count[0] &&
         taskData->outputs_count[1] == taskData->inputs_count[3];
}

bool SparseTBBMatrixMultiParallel::run() {
  internal_order_test();

  values3.clear();
  rows3.clear();
  colPtr3.clear();

  values3.resize(numCols2 * numRows1);  // Выделение памяти заранее для values3
  rows3.reserve(numCols2 * numRows1);   // Резервирование места в rows3
  colPtr3.reserve(numCols2 + 1);        // Резервирование места в colPtr3

  tbb::parallel_for(0, numCols1, [&](int j) {
    for (int k = colPtr2[j]; k < colPtr2[j + 1]; k++) {
      int column2 = j;
      int row2 = rows2[k];
      for (int l = colPtr1[row2]; l < colPtr1[row2 + 1]; l++) {
        int row1 = rows1[l];
        double val1 = values1[l];
        double val2 = values2[k];
        int index = row1 * numCols2 + column2;
        result[index] += val1 * val2;
      }
    }
  });

  tbb::parallel_for(0, numCols2, [&](int j) {
    colPtr3.push_back(values3.size());
    for (int i = 0; i < numRows1; i++) {
      int ind = i * numCols2 + j;
      if (result[ind] != 0.0) {
        values3[ind] = result[ind];
        rows3.push_back(i);
      }
    }
  });

  colPtr3.push_back(values3.size());
  return true;
}

bool SparseTBBMatrixMultiParallel::post_processing() {
  internal_order_test();

  auto* out_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  for (int i = 0; i < numRows3 * numCols3; i++) {
    out_ptr[i] = result[i];
  }

  delete[] result;

  return true;
}