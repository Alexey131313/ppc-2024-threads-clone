// Copyright 2024 Kulagin Aleksandr
#include "stl/kulagin_a_gauss_filter_vert/include/ops_stl.hpp"

#include <cstring>
#include <exception>
#include <thread>

enum class CURRENT_POSITION { START, END, MIDDLE };

bool FilterGaussVerticalTaskSTLKulagin::pre_processing() {
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

bool FilterGaussVerticalTaskSTLKulagin::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count.size() == 2 && taskData->inputs.size() == 2 && taskData->outputs.size() == 1 &&
         taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count == taskData->outputs_count && taskData->inputs[0] != nullptr &&
         taskData->inputs[1] != nullptr;
}

bool FilterGaussVerticalTaskSTLKulagin::run() {
  internal_order_test();
  try {
    auto thread_test = [this](size_t start, size_t end, const CURRENT_POSITION& pos) -> void {
      end--;
      switch (pos) {
        case CURRENT_POSITION::START:
          kulagin_a_gauss::apply_filter_line<true>(w, h, img, kernel, img_res.get(), start);
          start++;
        default:
          for (size_t x = start; x < end; x++) {
            kulagin_a_gauss::apply_filter_line<false>(w, h, img, kernel, img_res.get(), x);
          }
          break;
      }
      switch (pos) {
        case CURRENT_POSITION::END:
          kulagin_a_gauss::apply_filter_line<true>(w, h, img, kernel, img_res.get(), end);
          break;
        default:
          kulagin_a_gauss::apply_filter_line<false>(w, h, img, kernel, img_res.get(), end);
          break;
      }
    };
    const size_t max_threads = std::thread::hardware_concurrency();
    // if we have too many threads or 1 thread (also accounts for one edge case when w == 1)
    if (max_threads > w || max_threads <= 1) {
      kulagin_a_gauss::apply_filter(w, h, img, kernel, img_res.get());
      return true;
    }
    std::vector<std::thread> my_threads(max_threads);
    const size_t delta = w / max_threads;
    const size_t delta_left = w % max_threads;
    my_threads[0] = std::thread(thread_test, 0, delta, CURRENT_POSITION::START);
    size_t i;
    for (i = 1; i < max_threads - 1; i++) {
      my_threads[i] = std::thread(thread_test, i * delta, (i + 1) * delta, CURRENT_POSITION::MIDDLE);
    }
    my_threads[i] = std::thread(thread_test, i * delta, (i + 1) * delta + delta_left, CURRENT_POSITION::END);
    for (i = 0; i < max_threads; i++) {
      my_threads[i].join();
    }
  } catch (std::exception& e) {
    std::cout << e.what() << '\n';
    return false;
  } catch (...) {
    std::cout << "Something went very terrible!\n";
    return false;
  }
  return true;
}

bool FilterGaussVerticalTaskSTLKulagin::post_processing() {
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
