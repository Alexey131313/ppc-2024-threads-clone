// Copyright 2023 Kruglov Alexey

#include "seq/kruglov_a_components_marking/include/ops_seq.hpp"

using namespace std::chrono_literals;

bool imgMarkingSeq::pre_processing() {
  internal_order_test();
  // Init value for input and output
  h = reinterpret_cast<uint32_t *>(taskData->inputs[0])[0];
  w = reinterpret_cast<uint32_t *>(taskData->inputs[0])[1];
  src.resize(h);
  dst.resize(h);
  for (size_t i = 0; i < h; ++i) {
    for (size_t j = 0; j < w; ++j) src[i].push_back(reinterpret_cast<uint8_t *>(taskData->inputs[1])[i * w + j]);
    dst[i].resize(w, 0);
  }
  return true;
}

bool imgMarkingSeq::validation() {
  internal_order_test();
  // Check count elements of output
  h = reinterpret_cast<uint32_t *>(taskData->inputs[0])[0];
  w = reinterpret_cast<uint32_t *>(taskData->inputs[0])[1];
  return (h * w == taskData->inputs_count[1] && taskData->inputs_count[1] == taskData->outputs_count[0]);
}

bool imgMarkingSeq::run() {
  internal_order_test();
  imgMarking();
  return true;
}

bool imgMarkingSeq::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < h; ++i)
    for (size_t j = 0; j < w; ++j) reinterpret_cast<uint32_t *>(taskData->outputs[0])[i * w + j] = dst[i][j];
  return true;
}

void imgMarkingSeq::imgMarking() {
  uint32_t curLabel = 0;
  // std::vector<imgMarkingSeq::labelStore> labels;
  std::list<uint32_t> labelVec;
  std::vector<std::vector<uint32_t *>> ptrMap;
  ptrMap.resize(h);
  for (size_t i = 0; i < h; ++i) ptrMap[i].resize(w, nullptr);
  for (size_t i = 0; i < w; ++i) {
    if (src[0][i] == 0) {
      if (i == 0 || ptrMap[0][i - 1] == nullptr) {
        labelVec.push_back(++curLabel);
        ptrMap[0][i] = &labelVec.back();
      } else
        ptrMap[0][i] = ptrMap[0][i - 1];
    }
  }
  for (size_t i = 1; i < h; ++i) {
    if (src[i][0] == 0) {
      if (ptrMap[i - 1][0] == nullptr) {
        labelVec.push_back(++curLabel);
        ptrMap[i][0] = &labelVec.back();
      } else
        ptrMap[i][0] = ptrMap[i - 1][0];
    }
    for (size_t j = 1; j < w; ++j) {
      if (src[i][j] == 0) {
        if (ptrMap[i - 1][j] == ptrMap[i][j - 1]) {
          if (ptrMap[i - 1][j] == nullptr) {
            labelVec.push_back(++curLabel);
            ptrMap[i][j] = &labelVec.back();
          } else
            ptrMap[i][j] = ptrMap[i][j - 1];
        } else {
          if (ptrMap[i - 1][j] == nullptr)
            ptrMap[i][j] = ptrMap[i][j - 1];
          else if (ptrMap[i][j - 1] == nullptr)
            ptrMap[i][j] = ptrMap[i - 1][j];
          else {
            *(ptrMap[i][j - 1]) = *(ptrMap[i - 1][j]);
            ptrMap[i][j] = ptrMap[i - 1][j];
          }
        }
      }
    }
  }

  size_t count = 0;
  size_t cur = 0;
  labelVec.sort();
  for (auto &label : labelVec) {
    if (cur != label) {
      cur = label;
      count++;
    }
    label = count;
  }

  for (size_t i = 0; i < h; ++i)
    for (size_t j = 0; j < w; ++j)
      if (ptrMap[i][j] != nullptr) dst[i][j] = *(ptrMap[i][j]);
}