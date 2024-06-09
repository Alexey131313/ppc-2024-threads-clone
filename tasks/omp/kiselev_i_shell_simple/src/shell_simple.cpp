// Copyright 2024 Kiselev Igor
#include "omp/kiselev_i_shell_simple/include/shell_simple.hpp"

#include <omp.h>

#include <memory>

using namespace std::chrono_literals;

using namespace std;
using namespace Kiselev_omp;

bool KiselevTaskOMP::pre_processing() {
  try {
    internal_order_test();
    size_t n = taskData->inputs_count[0];
    arr = ::std::vector<int>(n, 0);
    for (size_t i = 0; i < n; ++i) {
      int *elem = reinterpret_cast<int *>(taskData->inputs[0] + i * sizeof(int));
      arr[i] = *elem;
    }
  } catch (...) {
    return false;
  }
  return true;
}

bool KiselevTaskOMP::validation() {
  try {
    internal_order_test();
    return taskData->inputs_count[0] != 0 && taskData->inputs_count[0] == taskData->outputs_count[0];
  } catch (...) {
    return false;
  }
}

bool KiselevTaskOMP::run() {
  try {
    internal_order_test();
    int n = (int)arr.size();
    FindThreadVariables();
    //  if (ThreadNum == 0) return false;
    int *Index = new int[ThreadNum + 1];
    int *BlockSize = new int[ThreadNum];
    int *BlockIndices = new int[ThreadNum];
    for (int i = 0; i <= ThreadNum; i++) {
      Index[i] = (i * n) / ThreadNum;
    }
    for (int i = 0; i < ThreadNum; i++) {
      BlockSize[i] = Index[i + 1] - Index[i];
      BlockIndices[i] = i;
    }
#pragma omp parallel for
    for (int i = 0; i < ThreadNum; i++) {
      SeqSorter(Index[i], Index[i] + BlockSize[i]);
    }
    for (int i = 1; i < ThreadNum; i *= 2) {
#pragma omp parallel for
      for (int j = 0; j < ThreadNum; j += 2 * i) {
        int left = BlockIndices[j];
        int right = (j + i < ThreadNum) ? BlockIndices[j + i] : -1;
        if (right != -1) {
          MergeBlocks(Index[left], BlockSize[left], Index[right], BlockSize[right]);
          BlockIndices[j] = left;
          BlockSize[j] = BlockSize[left] + BlockSize[right];
        } else {
          BlockIndices[j] = left;
          BlockSize[j] = BlockSize[left];
        }
      }
    }
    delete[] Index;
    delete[] BlockSize;
    delete[] BlockIndices;
    return true;
  } catch (...) {
    return false;
  }
  return true;
}

bool KiselevTaskOMP::post_processing() {
  try {
    internal_order_test();
    size_t n = arr.size();
    for (size_t i = 0; i < n; i++) {
      int *res = reinterpret_cast<int *>(taskData->outputs[0] + i * sizeof(int));
      *res = arr[i];
      (void)res;
    }
    return true;
  } catch (...) {
    return false;
  }
}
// Can do better
void KiselevTaskOMP::MergeBlocks(int Index1, int BlockSize1, int Index2, int BlockSize2) {
  int *pTempArray = new int[BlockSize1 + BlockSize2];
  for (int i = BlockSize1 + BlockSize2 - 1; i >= 0; i--) {
    pTempArray[i] = 0;
  }
  int i1 = Index1;
  int i2 = Index2;
  int curr = 0;
  if (BlockSize1 == 0) {
    for (int i = 0; i < BlockSize2; i++) {
      pTempArray[curr++] = arr[i2++];
    }
  } else if (BlockSize2 == 0) {
    for (int i = 0; i < BlockSize1; i++) {
      pTempArray[curr++] = arr[i1++];
    }
  } else {
    while (i1 < Index1 + BlockSize1 && i2 < Index2 + BlockSize2) {
      if (arr[i1] < arr[i2])
        pTempArray[curr++] = arr[i1++];
      else
        pTempArray[curr++] = arr[i2++];
    }
    while (i1 < Index1 + BlockSize1) pTempArray[curr++] = arr[i1++];
    while (i2 < Index2 + BlockSize2) pTempArray[curr++] = arr[i2++];
  }
  for (int i = 0; i < BlockSize1 + BlockSize2; i++) arr[Index1 + i] = pTempArray[i];
  delete[] pTempArray;
}

bool KiselevTaskOMP::IsSorted() {
  int n = arr.size();
  for (int i = 1; i < n; i++) {
    if (arr[i - 1] > arr[i]) return false;
  }
  return true;
}

int KiselevTaskOMP::exp(int arg, int exp) {
  int res = arg;
  for (int i = 1; i < exp; i++) {
    res *= arg;
  }
  return res;
}

void KiselevTaskOMP::FindThreadVariables() {
#pragma omp parallel
  {
#pragma omp single
    ThreadNum = omp_get_num_threads();
  }
  int helper = 1;
  DimSize = 1;
  while (ThreadNum >= helper) {
    DimSize++;
    helper = helper << 1;
  }
}

void KiselevTaskOMP::SeqSorter(int start, int end) {
  int n = end - start;
  for (int i = 1; i < n; i++) {
    int j = i - 1;
    while (j >= 0 && arr[j + start] > arr[j + 1 + start]) {
      arr[j + start] += arr[j + 1 + start];
      arr[j + 1 + start] = arr[j + start] - arr[j + 1 + start];
      arr[j + start] = arr[j + start] - arr[j + 1 + start];
      j--;
    }
  }
}
