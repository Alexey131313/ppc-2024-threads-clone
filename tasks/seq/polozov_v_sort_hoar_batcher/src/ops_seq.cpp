// Copyright 2024 Polozov Vladislav
#include "seq/polozov_v_sort_hoar_batcher/include/ops_seq.hpp"

bool SortHoarWithMergeBatcher::pre_processing() {
  internal_order_test();
  // Init value for input and output
  arr = std::vector<int>(taskData->inputs_count[0]);

  auto* temp_arr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    arr[i] = temp_arr[i];
  }
  return true;
}

bool SortHoarWithMergeBatcher::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool SortHoarWithMergeBatcher::run() {
  internal_order_test();
  res = odd_even_merge_with_hoar(arr);
  return true;
}

bool SortHoarWithMergeBatcher::post_processing() {
  internal_order_test();
  std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}

int partition(std::vector<int>& a, int l, int r) {
  int pivot = a[r];
  int m = l;
  for (int i = l; i <= r; i++) {
    if (a[i] <= pivot) {
      std::swap(a[i], a[m]);
      m++;
    }
  }
  return m - 1;
}

void Hoar_sort(std::vector<int>& a, int l, int r) {
  if (r <= l) return;
  int m = partition(a, l, r);
  Hoar_sort(a, l, m - 1);
  Hoar_sort(a, m + 1, r);
}

std::vector<int> generate_data(int n, int mn, int mx) {
  std::random_device rnd;
  std::default_random_engine generate{rnd()};
  std::uniform_real_distribution<double> random(mn, mx);
  std::vector<int> ans(n);
  for (int i = 0; i < n; i++) {
    ans[i] = random(generate);
  }
  return ans;
}

void CompAndSwap(int& x, int& y) {
  if (x > y) std::swap(x, y);
}

std::vector<int> odd_even_merge_with_hoar(std::vector<int>& my_data) {
  if (my_data.size() < 4) {
    Hoar_sort(my_data, 0, my_data.size() - 1);
    std::vector<int> ans = my_data;
    sort(ans.begin(), ans.end());
    return ans;
  }
  int n = my_data.size();
  std::vector<int> a(n / 2);
  std::vector<int> b(n / 2);
  for (int i = 0; i < n / 2; i++) {
    a[i] = my_data[i];
    b[i] = my_data[i + n / 2];
  }
  Hoar_sort(a, 0, n / 2 - 1);
  Hoar_sort(b, 0, n / 2 - 1);
  std::vector<int> ans(n);
  for (int i = 0; i < n / 2; i++) {
    ans[i] = a[i];
    ans[i + n / 2] = b[i];
  }
  for (int i = 0; i < n / 2; i++) {
    CompAndSwap(ans[i], ans[n - i - 1]);
  }
  for (int k = n / 2; k >= 2; k /= 2) {
    for (int i = 0; i < n / k; i++) {
      for (int j = 0; j < k / 2; j++) {
        CompAndSwap(ans[k * i + j], ans[k * i + j + k / 2]);
      }
    }
  }
  return ans;
}
