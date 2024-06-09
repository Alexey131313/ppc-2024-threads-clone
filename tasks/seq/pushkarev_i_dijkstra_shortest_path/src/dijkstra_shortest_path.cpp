// Copyright 2024 Pushkarev Ilya
#include "seq/pushkarev_i_dijkstra_shortest_path/include/dijkstra_shortest_path.hpp"

#include <limits>
#include <memory>
#include <thread>

bool DijkstraTask::pre_processing() {
  internal_order_test();
  graph = *reinterpret_cast<std::vector<std::vector<int>>*>(taskData->inputs[0]);
  source = *reinterpret_cast<int*>(taskData->inputs[1]);
  distances_ = *reinterpret_cast<std::vector<int>*>(taskData->outputs[0]);
  return true;
}

bool DijkstraTask::validation() {
  internal_order_test();
  std::vector<std::vector<int>> tmp_graph = *reinterpret_cast<std::vector<std::vector<int>>*>(taskData->inputs[0]);
  for (size_t i = 0; i < tmp_graph.size(); i++) {
    for (size_t j = 0; j < tmp_graph.size(); j++) {
      if (tmp_graph[i][j] < 0) {
        return false;
      }
    }
  }
  return true;
}

bool DijkstraTask::run() {
  internal_order_test();
  size_t n = distances_.size();
  for (size_t i = 0; i < n; i++) {
    distances_[i] = std::numeric_limits<int>::max();
  }
  distances_[source] = 0;
  std::vector<bool> processed(n, false);

  for (size_t count = 0; count < n; count++) {
    size_t u = getMinDistanceVertex(processed);
    processed[u] = true;
    for (size_t v = 0; v < n; v++) {
      if (!processed[v] && (graph[u][v] != 0) && (distances_[u] != std::numeric_limits<int>::max()) &&
          distances_[u] + graph[u][v] < distances_[v]) {
        relaxVertex(u, v);
      }
    }
  }
  return true;
}

bool DijkstraTask::post_processing() {
  internal_order_test();

  *reinterpret_cast<std::vector<int>*>(taskData->outputs[0]) = distances_;
  return true;
}

size_t DijkstraTask::getMinDistanceVertex(const std::vector<bool>& processed) {
  size_t min_dist = std::numeric_limits<size_t>::max();
  size_t min_index = 0;

  for (size_t v = 0; v < distances_.size(); v++) {
    if (!processed[v] && (size_t)distances_[v] <= min_dist) {
      min_dist = distances_[v];
      min_index = v;
    }
  }
  return min_index;
}

void DijkstraTask::relaxVertex(size_t u, size_t v) {
  distances_[v] = std::min(distances_[v], distances_[u] + graph[u][v]);
}
