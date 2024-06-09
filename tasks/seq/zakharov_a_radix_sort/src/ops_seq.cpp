// Copyright 2024 Zakharov Artem
#include "seq/zakharov_a_radix_sort/include/ops_seq.hpp"

bool ZakharovRadixSortSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool ZakharovRadixSortSequential::pre_processing() {
  internal_order_test();
  try {
    arr_size = taskData->inputs_count[0];
    auto* inp = reinterpret_cast<Number*>(taskData->inputs[0]);
    inp_arr.resize(arr_size);
    copy_data(inp, inp_arr.data(), arr_size);
    out_arr = reinterpret_cast<Number*>(taskData->outputs[0]);
    return true;
  } catch (...) {
    return false;
  }
}

bool ZakharovRadixSortSequential::run() {
  internal_order_test();
  try {
    Number* inp_arr_p = inp_arr.data();
    Number* out_arr_p = out_arr;
    for (size_t i = 0; i < num_bytes; i++) {
      counting_sort_by_bytes(inp_arr_p, out_arr_p, i);
      std::swap(inp_arr_p, out_arr_p);
    }
    return true;
  } catch (...) {
    return false;
  }
}

bool ZakharovRadixSortSequential::post_processing() {
  internal_order_test();
  try {
    copy_data(inp_arr.data(), out_arr, arr_size);
    return true;
  } catch (...) {
    return false;
  }
}

size_t ZakharovRadixSortSequential::get_index(const Number* const arr, std::size_t arr_ind, std::size_t byte_num) {
  if (byte_num == num_bytes - 1) {
    return static_cast<int8_t>(arr[arr_ind].bytes[byte_num]) + 128;
  }
  return arr[arr_ind].bytes[byte_num];
}

void ZakharovRadixSortSequential::counting_sort_by_bytes(const Number* const src, Number* const dest,
                                                         std::size_t byte_num) {
  std::fill(counter.begin(), counter.end(), 0);

  // count numbers
  for (std::size_t i = 0; i < arr_size; i++) {
    std::size_t ind = get_index(src, i, byte_num);
    counter[ind]++;
  }

  // generate offsets
  std::size_t prev = 0;
  for (auto& el : counter) {
    std::swap(prev, el);
    prev += el;
  }

  // sort elements
  for (std::size_t i = 0; i < arr_size; i++) {
    std::size_t ind = get_index(src, i, byte_num);
    dest[counter[ind]++].number = src[i].number;
  }
}

void ZakharovRadixSortSequential::copy_data(const Number* const src, Number* const dist, std::size_t size) {
  for (std::size_t i = 0; i < size; i++) {
    dist[i].number = src[i].number;
  }
}
