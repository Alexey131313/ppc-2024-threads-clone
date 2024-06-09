// Copyright 2024 Makhinya Danil

#include "seq/makhinya_d_hoare_sort_batcher_merge/include/hoare_sort.hpp"

bool HoareSort::pre_processing() {
  internal_order_test();
  // Init value for input and output
  _data = reinterpret_cast<vec_t*>(taskData->inputs[0]);
  if (taskData->inputs.size() == 2) {
    _comp = *(reinterpret_cast<compare_t*>(taskData->inputs[1]));
  } else {
    _comp = default_compare;
  }
  return true;
}

bool HoareSort::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs.size() == 1 || taskData->inputs.size() == 2) && taskData->outputs.size() <= 1;
}

bool HoareSort::run() {
  internal_order_test();
  sortable_type* begin = _data->data();
  sortable_type* end = begin + _data->size() - 1UL;
  seq_hoare_sort(begin, end);
  return true;
}

bool HoareSort::post_processing() {
  internal_order_test();
  return true;
}

bool HoareSort::check_order() {
  for (uint32_t i = 1U; i < _data->size(); ++i) {
    if (_comp((*_data)[i], (*_data)[i - 1U])) {
      return false;
    }
  }

  return true;
}

void HoareSort::seq_hoare_sort(sortable_type* first_ptr, sortable_type* last_ptr) {
  if (last_ptr <= first_ptr) {
    return;
  }

  sortable_type m = *(first_ptr + (last_ptr + 1UL - first_ptr) / 2UL);
  sortable_type* ll = first_ptr;
  sortable_type* rr = last_ptr;
  while (ll <= rr) {
    while (HoareSort::_comp(*ll, m)) ll++;
    while (HoareSort::_comp(m, *rr)) rr--;

    if (ll <= rr) {
      std::swap(*ll, *rr);
      ll++;
      rr--;
    }
  }

  if (first_ptr < rr) {
    HoareSort::seq_hoare_sort(first_ptr, rr);
  }
  if (ll < last_ptr) {
    HoareSort::seq_hoare_sort(ll, last_ptr);
  }
}
