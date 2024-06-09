// Copyright 2024 Lesnikov Nikita
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class BinaryLabellingSequential : public ppc::core::Task {
 public:
  explicit BinaryLabellingSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static std::vector<uint8_t> serializeInt32(uint32_t num);
  static uint32_t deserializeInt32(const uint8_t* data);
  static void visualize(std::vector<uint8_t>& v, int m, int n);

 private:
  static uint8_t& _get(std::vector<uint8_t>& v, int n, int x, int y);
  static bool _inBounds(int x, int y, int m, int n);
  static bool _notPrev(int x, int y, int prevX, int prevY);
  static bool _markObject(std::vector<uint8_t> v, std::vector<uint8_t>& labelled, std::vector<uint8_t>& checked, int m,
                          int n, int x, int y, int prevX, int prevY, int label);
  static std::pair<std::vector<uint8_t>, int> _getLabelledImage(const std::vector<uint8_t>& v, int m, int n);

  std::vector<uint8_t> _source;
  std::vector<uint8_t> _result;
  uint32_t _m = 0;
  uint32_t _n = 0;
  uint32_t _numObjects = 0;
};
