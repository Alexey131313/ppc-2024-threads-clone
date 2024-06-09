//
// Created by Стёпа on 29.03.2024.
//

#include "seq/kashin_s_dijkstra_algorithm/include/Dijkstra.hpp"

bool KashinDijkstraSeq::Dijkstra::pre_processing() {
  internal_order_test();
  graph = reinterpret_cast<int*>(taskData->inputs[0]);
  count = taskData->outputs_count[0];
  start = taskData->inputs_count[1];
  distance = std::vector(count, INT_MAX);
  distance[start] = 0;
  return true;
}

bool KashinDijkstraSeq::Dijkstra::validation() {
  internal_order_test();
  int countg = taskData->inputs_count[0];
  int countd = taskData->outputs_count[0];
  int startm = taskData->inputs_count[1];
  if (countg / countd != countd) {
    return false;
  }
  if (startm < 0 || startm >= countd) {
    return false;
  }
  if (countg < 0 || countd < 0) {
    return false;
  }

  return true;
}

bool KashinDijkstraSeq::Dijkstra::run() {
  internal_order_test();
  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, KashinDijkstraSeq::Compare> pq;
  pq.emplace(0, start);
  while (!pq.empty()) {
    std::pair<int, int> vertex = pq.top();
    pq.pop();

    for (int i = 0; i < count; i++) {
      std::pair<int, int> edge(graph[vertex.second * count + i], i);
      if (i != vertex.second && edge.first != -1) {
        int weight = edge.first + vertex.first;
        if (weight < distance[edge.second]) {
          distance[edge.second] = weight;
          pq.emplace(weight, edge.second);
        }
      }
    }
  }
  return true;
}

bool KashinDijkstraSeq::Dijkstra::post_processing() {
  internal_order_test();
  int* out_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  for (uint32_t i = 0; i < distance.size(); i++) {
    out_ptr[i] = distance[i];
  }
  return true;
}
