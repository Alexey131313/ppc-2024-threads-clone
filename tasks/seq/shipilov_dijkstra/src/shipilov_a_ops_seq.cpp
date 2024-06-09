// Copyright 2024 Shipilov Artem
#include "seq/shipilov_dijkstra/include/shipilov_a_ops_seq.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

using namespace Dijkstra;

void graph_t::add_edge(const node_id_t &src, const edge_t &new_edge) {
  auto target = new_edge.get_target();
  if (target >= _graph.size()) {
    _graph.resize(target + 1);
  }

  _graph[src].push_back(new_edge);
}

void graph_t::normalize() {
  if (_graph.empty()) {
    return;
  }

  for (auto &node_data : _graph) {
    if (node_data.empty()) {
      continue;
    }

    auto insert_it = node_data.begin();

    for (auto curr_it = std::next(node_data.begin()); curr_it < node_data.end(); ++curr_it) {
      if (curr_it->get_target() != insert_it->get_target()) {
        *(++insert_it) = *curr_it;
      }
    }

    node_data.erase(std::next(insert_it), node_data.end());
  }
}

std::vector<path_data_t> graph_t::calculate_paths(const node_id_t &start_node) {
  normalize();

  const size_t graph_size = _graph.size();
  std::vector<bool> calculated(graph_size);
  std::vector<path_data_t> result(graph_size, path_data_t{invalid_node_id, std::numeric_limits<weight_t>::max()});

  result[start_node].parent = start_node;
  result[start_node].summary_dist = 0;
  calculated[start_node] = true;

  node_id_t current_node = start_node;

  for (size_t i = 0; i < graph_size; i++) {
    const auto &node_edges = _graph[current_node];
    const size_t sz = node_edges.size();
    auto current_weight = result[current_node].summary_dist;
    for (size_t j = 0; j < sz; j++) {
      const auto &edge = node_edges[j];
      auto dst = edge.get_target();
      auto w = edge.get_weight();
      if (result[dst].summary_dist > current_weight + w) {
        result[dst].summary_dist = current_weight + w;
        result[dst].parent = current_node;
      }
    }

    node_id_t nearest_node{};
    weight_t minimal_dist = std::numeric_limits<weight_t>::max();
    for (size_t k = 0; k < graph_size; k++) {
      if (!calculated[k] && result[k].summary_dist < minimal_dist) {
        minimal_dist = result[k].summary_dist;
        nearest_node = k;
      }
    }

    calculated[nearest_node] = true;
    current_node = nearest_node;
  }

  return result;
}

bool MyDijkstra::pre_processing() {
  internal_order_test();
  return true;
}

bool MyDijkstra::validation() {
  internal_order_test();
  return true;
}

bool MyDijkstra::run() {
  internal_order_test();
  out = graph.calculate_paths(0);
  return true;
}

bool MyDijkstra::post_processing() {
  internal_order_test();
  return true;
}
