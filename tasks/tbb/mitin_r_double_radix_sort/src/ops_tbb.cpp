// Copyright 2024 Mitin Roman
#include "tbb/mitin_r_double_radix_sort/include/ops_tbb.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#include "tbb/parallel_for.h"

using namespace std::chrono_literals;
using namespace mitinr_radix_sort;

namespace {

constexpr int64_t max_byte_val = 256;

uint8_t get_byte_val(double val, size_t byte_num) {
  assert(*(reinterpret_cast<uint8_t*>(&val) + byte_num) < max_byte_val);
  return *(reinterpret_cast<uint8_t*>(&val) + byte_num);
}

void sort_arr(double* arr, size_t size, double* buckets) {
  if (size != 0) {
    for (size_t byte_num = 0; byte_num < 7; byte_num++) {
      size_t buckets_size[max_byte_val]{0};
      for (size_t i = 0; i < size; i++) {
        const uint32_t byte_val = get_byte_val(arr[i], byte_num);
        buckets[byte_val * size + buckets_size[byte_val]++] = arr[i];
      }

      size_t it = 0;
      for (int64_t i = 0; i < max_byte_val; i++) {
        if (buckets_size[i] != 0) {
          memcpy(arr + it, &buckets[i * size], buckets_size[i] * sizeof(double));
          it += buckets_size[i];
          assert(it <= size);
        }
      }
    }
  }
}

}  // namespace

bool SortRadixDoubleTaskTBB::validation() {
  internal_order_test();
  // Check count elements of output
  return !taskData->inputs_count.empty() || taskData->inputs_count.size() == taskData->outputs_count.size() ||
         taskData->outputs[0] != nullptr || taskData->inputs_count[0] == 0 || !taskData->inputs.empty() ||
         taskData->inputs[0] != nullptr;
}

bool SortRadixDoubleTaskTBB::pre_processing() {
  internal_order_test();
  // Init value for input and output

  data_size = taskData->inputs_count[0];
  data_ptr = reinterpret_cast<double*>(taskData->inputs[0]);

  return data_size == 0 || data_ptr != nullptr;
}

bool SortRadixDoubleTaskTBB::run() {
  internal_order_test();

  // tbb::task_scheduler_init init;

  try {
    constexpr int64_t first_byte_max_val = 128;
    auto* buckets = reinterpret_cast<double*>(std::malloc(data_size * first_byte_max_val * sizeof(double)));
    if (buckets == nullptr) {
      std::cerr << "Cannot allocate buffer\n";
      return false;
    }
    size_t buckets_size[first_byte_max_val]{0};

    uint64_t max_bucket_size = 0;
    for (size_t i = 0; i < data_size; i++) {
      const uint32_t byte_val = get_byte_val(data_ptr[i], 7);
      assert(byte_val < first_byte_max_val);
      buckets[byte_val * data_size + buckets_size[byte_val]++] = data_ptr[i];
      max_bucket_size = std::max<size_t>(max_bucket_size, buckets_size[byte_val]);
    }

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, first_byte_max_val), [=, this](const tbb::blocked_range<size_t>& r) {
          auto* local_buckets = reinterpret_cast<double*>(malloc(max_bucket_size * max_byte_val * sizeof(double)));
          for (size_t i = r.begin(); i < r.end(); i++) {
            sort_arr(&buckets[i * data_size], buckets_size[i], local_buckets);
          };
          free(local_buckets);
        });

    size_t it = 0;
    for (int64_t i = 0; i < first_byte_max_val; i++) {
      if (buckets_size[i] != 0) {
        std::memcpy(data_ptr + it, &buckets[i * data_size], buckets_size[i] * sizeof(double));
        it += buckets_size[i];
        assert(it <= data_size);
      }
    }

    free(buckets);

  } catch (const std::exception& e) {
    std::cerr << "Double Radix sort Exception error: " << e.what() << std::endl;
    return false;
  }

  return true;
}

bool SortRadixDoubleTaskTBB::post_processing() {
  internal_order_test();
  *reinterpret_cast<double**>(taskData->outputs[0]) = data_ptr;
  return true;
}
