// Copyright 2024 Petrov Maksim

#include "omp/petrov_m_radix_sort_double/include/ops_omp.hpp"

#include <omp.h>

#include <thread>
#include <vector>

using namespace std::chrono_literals;

void PetrovRadixSortDoubleOMP::PetrovCountSort(double* in, double* out, int len, int exp) {
  auto* buf = reinterpret_cast<unsigned char*>(in);
  int count[256] = {0};
  for (int i = 0; i < len; i++) {
    count[buf[8 * i + exp]]++;
  }
  int sum = 0;
  for (int i = 0; i < 256; i++) {
    int temp = count[i];
    count[i] = sum;
    sum += temp;
  }
  for (int i = 0; i < len; i++) {
    out[count[buf[8 * i + exp]]] = in[i];
    count[buf[8 * i + exp]]++;
  }
}

bool PetrovRadixSortDoubleOMP::PetrovCountSortSigns(const double* in, double* out, int len) {
  bool positiveFlag = false;
  bool negativeFlag = false;
  int firstNegativeIndex = -1;
  // int firstPositiveIndex = -1;
  for (int i = 0; i < len; i++) {
    if (positiveFlag && negativeFlag) {
      break;
    }
    if (in[i] < 0 && !negativeFlag) {
      negativeFlag = true;
      firstNegativeIndex = i;
    }
    if (in[i] > 0 && !positiveFlag) {
      positiveFlag = true;
      // firstPositiveIndex = i;
    }
  }
  if (positiveFlag && negativeFlag) {
    bool forward = false;
    int j = len - 1;
    for (int i = 0; i < len; i++) {
      out[i] = in[j];
      if (forward) {
        j++;
      } else {
        j--;
      }
      if (j == firstNegativeIndex - 1 && !forward) {
        j = 0;
        forward = true;
      }
    }
    return true;
  }
  if (!positiveFlag) {
    for (int i = len - 1, j = 0; i >= 0; i--, j++) {
      out[j] = in[i];
    }
    return true;
  }
  return false;
}

std::vector<std::vector<double>> PetrovRadixSortDoubleOMP::PetrovSplitVector(const std::vector<double>& data,
                                                                             int numParts) {
  std::vector<std::vector<double>> resultVec;
  std::vector<double> tmp;

  if (numParts < 2 || (int)data.size() < numParts) {
    resultVec.push_back(data);
    return resultVec;
  }

  int part = data.size() / numParts;
  int remainder = data.size() % numParts;

  for (int i = 0; i < numParts; ++i) {
    int start = i * part + std::min(i, remainder);
    int end = start + part + (i < remainder ? 1 : 0);
    resultVec.emplace_back(data.begin() + start, data.begin() + end);
  }

  return resultVec;
}

std::vector<double> PetrovRadixSortDoubleOMP::PetrovMerge(const std::vector<double>& arr1,
                                                          const std::vector<double>& arr2) {
  int len1 = arr1.size();
  int len2 = arr2.size();
  std::vector<double> out;
  out.reserve(len1 + len2);
  int indexFirst = 0;
  int indexSecond = 0;
  while (indexFirst < len1 && indexSecond < len2) {
    if (arr1[indexFirst] < arr2[indexSecond]) {
      out.push_back(arr1[indexFirst++]);
    } else {
      out.push_back(arr2[indexSecond++]);
    }
  }

  while (indexFirst < len1) {
    out.push_back(arr1[indexFirst++]);
  }
  while (indexSecond < len2) {
    out.push_back(arr2[indexSecond++]);
  }

  return out;
}

std::vector<double> PetrovRadixSortDoubleOMP::PetrovRadixSort(const std::vector<double>& data) {
  int len = static_cast<int>(data.size());
  std::vector<double> in = data;
  std::vector<double> out(data.size());

  for (int i = 0; i < 4; i++) {
    PetrovCountSort(in.data(), out.data(), len, 2 * i);
    PetrovCountSort(out.data(), in.data(), len, 2 * i + 1);
  }
  if (!PetrovCountSortSigns(in.data(), out.data(), len)) {
    in.swap(out);
  }
  return out;
}

std::vector<double> PetrovRadixSortDoubleOMP::PetrovBinaryMergeTree(std::vector<std::vector<double>>& sortedVectors) {
  while (sortedVectors.size() > 1) {
    std::vector<std::vector<double>> mergedVectors;

// In parallel, merge the vectors in pairs
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(sortedVectors.size()) - 1; i += 2) {
      std::vector<double> merged = PetrovMerge(sortedVectors[i], sortedVectors[i + 1]);
#pragma omp critical
      mergedVectors.push_back(std::move(merged));
    }

    // If the number of vectors is odd, just add the last vector to the result
    if (sortedVectors.size() % 2 != 0) {
      mergedVectors.push_back(std::move(sortedVectors.back()));
    }

    // Updating the list of sorted vectors for the next merge level
    sortedVectors.swap(mergedVectors);
  }

  return sortedVectors.front();
}

std::vector<double> PetrovRadixSortDoubleOMP::PetrovRadixSortOmp(const std::vector<double>& data, int numParts) {
  std::vector<std::vector<double>> vectorsForParallel = PetrovSplitVector(data, numParts);
  omp_set_num_threads(numParts);

#pragma omp parallel for
  for (int i = 0; i < (int)vectorsForParallel.size(); ++i) {
    vectorsForParallel[i] = PetrovRadixSort(vectorsForParallel[i]);
  }

  std::vector<double> finalResult = PetrovBinaryMergeTree(vectorsForParallel);
  return finalResult;
}

bool PetrovRadixSortDoubleOMP::pre_processing() {
  internal_order_test();
  try {
    data_size = taskData->inputs_count[0];
    while (!sort.empty()) {
      sort.pop_back();
    }
    auto* inp = reinterpret_cast<double*>((taskData->inputs[0]));
    for (int i = 0; i < data_size; i++) {
      sort.push_back(inp[i]);
    }
  } catch (...) {
    std::cout << "\n";
    std::cout << "Double radix sort error";
    std::cout << "\n";
    return false;
  }
  return true;
}

bool PetrovRadixSortDoubleOMP::validation() {
  internal_order_test();
  // Check count elements of output
  return ((taskData->inputs_count[0] > 1) && (taskData->outputs_count[0] == taskData->inputs_count[0]));
}

bool PetrovRadixSortDoubleOMP::run() {
  internal_order_test();
  try {
    sort = (PetrovRadixSortOmp(sort, 6));
  } catch (...) {
    std::cout << "\n";
    std::cout << "Double radix sort error";
    std::cout << "\n";
    return false;
  }
  return true;
}

bool PetrovRadixSortDoubleOMP::post_processing() {
  internal_order_test();
  try {
    auto* outputs = reinterpret_cast<double*>(taskData->outputs[0]);
    for (int i = 0; i < data_size; i++) {
      outputs[i] = sort[i];
    }
  } catch (...) {
    std::cout << "\n";
    std::cout << "Double radix sort error";
    std::cout << "\n";
    return false;
  }
  return true;
}
