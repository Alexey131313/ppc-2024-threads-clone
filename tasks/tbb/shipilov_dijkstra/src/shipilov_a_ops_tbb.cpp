// Copyright 2024 Shipilov Artem
#include "tbb/shipilov_dijkstra/include/shipilov_a_ops_tbb.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "tbb/tbb.h"

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

    // remove duplicates
    auto new_end_it = std::unique(node_data.begin(), node_data.end(),
                                  [&](auto a, auto b) { return a.get_target() == b.get_target(); });
    node_data.erase(new_end_it, node_data.end());
  }
}

std::vector<path_data_t> graph_t::calculate_paths_omp(const node_id_t &start_node) {
  normalize();

  const int graph_size = _graph.size();
  std::vector<bool> calculated(graph_size);
  std::vector<path_data_t> result(graph_size, path_data_t{invalid_node_id, (std::numeric_limits<weight_t>::max)()});

  result[start_node].parent = start_node;
  result[start_node].summary_dist = 0;
  node_id_t current_node = start_node;

  for (int i = 0; i < graph_size; i++) {
    calculated[current_node] = true;
    const auto &node_edges = _graph[current_node];
    const int sz = node_edges.size();
    auto current_weight = result[current_node].summary_dist;

    // parallel relaxation from current node
    tbb::parallel_for(0, sz, 1, [&](node_id_t j) {
      const auto &edge = node_edges[j];
      auto dst = edge.get_target();
      auto w = edge.get_weight();
      if (result[dst].summary_dist > current_weight + w) {
        result[dst].summary_dist = current_weight + w;
        result[dst].parent = current_node;
      }
    });

    // parallel calculation of nearest node
    auto block_argmin = [&](const tbb::blocked_range<node_id_t> &r, node_id_t node) {
      for (node_id_t k = r.begin(); k < r.end(); ++k) {
        if (!calculated[k] && (node == invalid_node_id || result[k].summary_dist < result[node].summary_dist)) {
          node = k;
        }
      }
      return node;
    };
    auto reduction = [&](const node_id_t x, const node_id_t y) {
      if (x == invalid_node_id) return y;
      if (y == invalid_node_id) return x;
      return result[x].summary_dist < result[y].summary_dist ? x : y;
    };
    node_id_t nearest_node =
        tbb::parallel_reduce(tbb::blocked_range<node_id_t>(0, graph_size), invalid_node_id, block_argmin, reduction);
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
  out = graph.calculate_paths_omp(0);
  return true;
}

bool MyDijkstra::post_processing() {
  internal_order_test();
  return true;
}
