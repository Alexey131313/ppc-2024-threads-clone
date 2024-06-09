// Copyright 2024 Shipilov Artem
#ifndef DIJKSTRA_HPP
#define DIJKSTRA_HPP

#include <cinttypes>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace Dijkstra {
using node_id_t = uint64_t;
static constexpr node_id_t invalid_node_id = (node_id_t)-1;
using weight_t = uint64_t;

class edge_t {
 public:
  edge_t(const node_id_t &target, const weight_t weight) noexcept : target(target), weight(weight) {}

  const node_id_t &get_target() const noexcept { return target; }

  weight_t get_weight() const noexcept { return weight; }

  bool operator<(const edge_t &oth) const noexcept {
    if (target == oth.target) {
      return weight < oth.weight;
    }

    return target < oth.target;
  }

  bool operator==(const edge_t &oth) const noexcept { return target == oth.target && weight == oth.weight; }

 private:
  node_id_t target;
  weight_t weight;
};

struct path_data_t {
  node_id_t parent = invalid_node_id;
  weight_t summary_dist{};

  bool operator==(const path_data_t &oth) const noexcept {
    return parent == oth.parent && summary_dist == oth.summary_dist;
  }
  bool operator!=(const path_data_t &oth) const noexcept {
    return parent == oth.parent && summary_dist == oth.summary_dist;
  }
};

class graph_t {
 public:
  graph_t() = default;

  graph_t(size_t node_number) : _graph(node_number) {}

  void add_edge(const node_id_t &src, const edge_t &new_edge);

  std::vector<path_data_t> calculate_paths_omp(const node_id_t &start_node);

 private:
  void normalize();

  std::vector<std::vector<edge_t>> _graph;
};

class MyDijkstra : public ppc::core::Task {
 public:
  explicit MyDijkstra(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  graph_t graph;
  std::vector<path_data_t> out;
};

}  // namespace Dijkstra

#endif  // DIJKSTRA_HPP
