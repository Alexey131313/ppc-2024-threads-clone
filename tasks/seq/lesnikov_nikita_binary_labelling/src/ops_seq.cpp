// Copyright 2024 Lesnikov Nikita
#include "seq/lesnikov_nikita_binary_labelling/include/ops_seq.hpp"

#include <cstring>
#include <thread>

using namespace std::chrono_literals;

bool BinaryLabellingSequential::pre_processing() {
  internal_order_test();
  try {
    _source.resize(taskData->inputs_count[0]);
    memcpy(_source.data(), taskData->inputs[0], taskData->inputs_count[0]);
    _m = deserializeInt32(taskData->inputs[1]);
    _n = deserializeInt32(taskData->inputs[2]);
  } catch (...) {
    return false;
  }

  return true;
}

bool BinaryLabellingSequential::validation() {
  internal_order_test();

  return taskData->inputs_count.size() == 3 && taskData->outputs_count.size() == 2 && taskData->inputs_count[1] == 4 &&
         taskData->inputs_count[2] == 4 && taskData->outputs_count[1] == 4 &&
         taskData->inputs_count[0] == deserializeInt32(taskData->inputs[1]) * deserializeInt32(taskData->inputs[2]);
}

bool BinaryLabellingSequential::run() {
  internal_order_test();
  try {
    auto res = _getLabelledImage(_source, _m, _n);
    _result = res.first;
    _numObjects = res.second;
  } catch (...) {
    return false;
  }
  return true;
}

bool BinaryLabellingSequential::post_processing() {
  internal_order_test();
  try {
    memcpy(taskData->outputs[0], _result.data(), _result.size());
    auto serializedObjectsNum = serializeInt32(_numObjects);
    memcpy(taskData->outputs[1], serializedObjectsNum.data(), serializedObjectsNum.size());
  } catch (...) {
    return false;
  }
  return true;
}

std::vector<uint8_t> BinaryLabellingSequential::serializeInt32(uint32_t num) {
  std::vector<uint8_t> result;
  for (int i = 3; i >= 0; i--) {
    result.push_back(static_cast<uint8_t>(((num << (3 - i) * 8) >> (3 - i) * 8) >> i * 8));
  }
  return result;
}

uint32_t BinaryLabellingSequential::deserializeInt32(const uint8_t* data) {
  uint32_t res = 0;
  for (int i = 3; i >= 0; i--) {
    res += static_cast<uint32_t>(data[3 - i]) << i * 8;
  }
  return res;
}

uint8_t& BinaryLabellingSequential::_get(std::vector<uint8_t>& v, int n, int x, int y) {
  return v[static_cast<size_t>(x * n + y)];
}

bool BinaryLabellingSequential::_inBounds(int x, int y, int m, int n) { return x >= 0 && x < m && y >= 0 && y < n; }

bool BinaryLabellingSequential::_notPrev(int x, int y, int prevX, int prevY) { return x != prevX || y != prevY; }

bool BinaryLabellingSequential::_markObject(std::vector<uint8_t> v, std::vector<uint8_t>& labelled,
                                            std::vector<uint8_t>& checked, int m, int n, int x, int y, int prevX,
                                            int prevY, int label) {
  if (_get(checked, n, x, y) != 0) {
    return false;
  }
  _get(checked, n, x, y) = 1;
  if (_get(v, n, x, y) == 0) {
    _get(labelled, n, x, y) = label;
  } else {
    return false;
  }
  if (_inBounds(x - 1, y, m, n) && _notPrev(x - 1, y, prevX, prevY)) {
    _markObject(v, labelled, checked, m, n, x - 1, y, x, y, label);
  }
  if (_inBounds(x + 1, y, m, n) && _notPrev(x + 1, y, prevX, prevY)) {
    _markObject(v, labelled, checked, m, n, x + 1, y, x, y, label);
  }
  if (_inBounds(x, y - 1, m, n) && _notPrev(x, y - 1, prevX, prevY)) {
    _markObject(v, labelled, checked, m, n, x, y - 1, x, y, label);
  }
  if (_inBounds(x, y + 1, m, n) && _notPrev(x, y + 1, prevX, prevY)) {
    _markObject(v, labelled, checked, m, n, x, y + 1, x, y, label);
  }
  return true;
}

std::pair<std::vector<uint8_t>, int> BinaryLabellingSequential::_getLabelledImage(const std::vector<uint8_t>& v, int m,
                                                                                  int n) {
  std::vector<uint8_t> labelled(v.size());
  std::vector<uint8_t> checked(v.size());
  int label = 1;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      label += static_cast<int>(_markObject(v, labelled, checked, m, n, i, j, -1, -1, label));
    }
  }
  return std::make_pair(labelled, label - 1);
}

void BinaryLabellingSequential::visualize(std::vector<uint8_t>& v, int m, int n) {
  std::cout << "\n\n";
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << (int)_get(v, n, i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}
