// Copyright 2024 Kulagin Aleksandr
#include "seq/kulagin_a_gauss_filter_vert/include/ops_seq.hpp"

#include <cstring>
#include <exception>

bool FilterGaussVerticalTaskSequential::pre_processing() {
  internal_order_test();
  try {
    // Init value for input and output
    img = reinterpret_cast<uint32_t*>(taskData->inputs[0]);
    kernel = reinterpret_cast<float*>(taskData->inputs[1]);
    w = taskData->inputs_count[0];
    h = taskData->inputs_count[1];
    img_res = std::make_unique<uint32_t[]>(w * h);
  } catch (std::exception& e) {
    std::cout << e.what() << '\n';
    return false;
  } catch (...) {
    std::cout << "Something went very terrible!\n";
    return false;
  }
  return true;
}

bool FilterGaussVerticalTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count.size() == 2 && taskData->inputs.size() == 2 && taskData->outputs.size() == 1 &&
         taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count == taskData->outputs_count && taskData->inputs[0] != nullptr &&
         taskData->inputs[1] != nullptr;
}

bool FilterGaussVerticalTaskSequential::run() {
  internal_order_test();
  try {
    kulagin_a_gauss::apply_filter(w, h, img, kernel, img_res.get());
  } catch (std::exception& e) {
    std::cout << e.what() << '\n';
    return false;
  } catch (...) {
    std::cout << "Something went very terrible!\n";
    return false;
  }
  return true;
}

bool FilterGaussVerticalTaskSequential::post_processing() {
  internal_order_test();
  try {
    std::memcpy(reinterpret_cast<uint32_t*>(taskData->outputs[0]), img_res.get(), w * h * sizeof(uint32_t));
  } catch (std::exception& e) {
    std::cout << e.what() << '\n';
    return false;
  } catch (...) {
    std::cout << "Something went very terrible!\n";
    return false;
  }
  return true;
}
