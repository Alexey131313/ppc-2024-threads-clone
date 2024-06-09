// Copyright 2024 Shipilov Artem
#include <gtest/gtest.h>

#include "tbb/shipilov_dijkstra/include/shipilov_a_ops_tbb.hpp"

TEST(Dijkstra_Shipilov_tbb_test, Test_Correct_Work_One) {
  // Create data
  std::vector<std::tuple<int, int, int> > in{{0, 1, 11}};
  std::vector<Dijkstra::path_data_t> out{{0, 0}, {0, 11}};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  size_t m = in.size();
  int n = 2;
  Dijkstra::graph_t graph(n);
  for (size_t i = 0; i < m; i++) {
    int u = std::get<0>(in[i]);
    int v = std::get<1>(in[i]);
    int w = std::get<2>(in[i]);
    graph.add_edge(u, Dijkstra::edge_t(v, w));
    graph.add_edge(v, Dijkstra::edge_t(u, w));
  };

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  Dijkstra::MyDijkstra testTaskSequential(taskDataSeq);
  testTaskSequential.graph = std::move(graph);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(testTaskSequential.out, out);
}

TEST(Dijkstra_Shipilov_tbb_test, Test_Correct_Work_Two) {
  // Create data
  std::vector<std::tuple<int, int, int> > in{{0, 1, 10}};
  std::vector<Dijkstra::path_data_t> out{
      {0, 0}, {0, 10}, {Dijkstra::invalid_node_id, std::numeric_limits<Dijkstra::weight_t>::max()}};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  size_t m = in.size();
  int n = 3;
  Dijkstra::graph_t graph(n);
  for (size_t i = 0; i < m; i++) {
    int u = std::get<0>(in[i]);
    int v = std::get<1>(in[i]);
    int w = std::get<2>(in[i]);

    graph.add_edge(u, Dijkstra::edge_t(v, w));
    graph.add_edge(v, Dijkstra::edge_t(u, w));
  };

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  Dijkstra::MyDijkstra testTaskSequential(taskDataSeq);
  testTaskSequential.graph = std::move(graph);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(testTaskSequential.out, out);
}

TEST(Dijkstra_Shipilov_tbb_test, Test_Correct_Work_Three) {
  // Create data
  std::vector<std::tuple<int, int, int> > in{{0, 1, 4}, {0, 3, 1}, {0, 2, 3}, {3, 2, 1}, {3, 4, 4}};
  std::vector<Dijkstra::path_data_t> out{{0, 0}, {0, 4}, {3, 2}, {0, 1}, {3, 5}};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  size_t m = in.size();
  int n = 5;
  Dijkstra::graph_t graph(n);
  for (size_t i = 0; i < m; i++) {
    int u = std::get<0>(in[i]);
    int v = std::get<1>(in[i]);
    int w = std::get<2>(in[i]);

    graph.add_edge(u, Dijkstra::edge_t(v, w));
    graph.add_edge(v, Dijkstra::edge_t(u, w));
  };

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  Dijkstra::MyDijkstra testTaskSequential(taskDataSeq);
  testTaskSequential.graph = std::move(graph);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(testTaskSequential.out, out);
}

TEST(Dijkstra_Shipilov_tbb_test, Test_Correct_Work_Four) {
  // Create data
  std::vector<std::tuple<int, int, int> > in{{0, 1, 10}, {0, 2, 5}, {2, 3, 4}, {2, 5, 7}, {5, 3, 1}};
  std::vector<Dijkstra::path_data_t> out{
      {0, 0}, {0, 10}, {0, 5}, {2, 9}, {Dijkstra::invalid_node_id, std::numeric_limits<Dijkstra::weight_t>::max()},
      {3, 10}};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  size_t m = in.size();
  int n = 6;
  Dijkstra::graph_t graph(n);
  for (size_t i = 0; i < m; i++) {
    int u = std::get<0>(in[i]);
    int v = std::get<1>(in[i]);
    int w = std::get<2>(in[i]);

    graph.add_edge(u, Dijkstra::edge_t(v, w));
    graph.add_edge(v, Dijkstra::edge_t(u, w));
  };

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  Dijkstra::MyDijkstra testTaskSequential(taskDataSeq);
  testTaskSequential.graph = std::move(graph);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(testTaskSequential.out, out);
}

TEST(Dijkstra_Shipilov_tbb_test, Test_Correct_Work_Five) {
  // Create data
  std::vector<std::tuple<int, int, int> > in{{0, 1, 11}, {0, 2, 10}, {0, 5, 1}, {2, 5, 5},
                                             {2, 3, 7},  {3, 4, 3},  {4, 5, 3}, {5, 6, 2}};
  std::vector<Dijkstra::path_data_t> out{
      {0, 0}, {0, 11}, {5, 6}, {4, 7},
      {5, 4}, {0, 1},  {5, 3}, {Dijkstra::invalid_node_id, std::numeric_limits<Dijkstra::weight_t>::max()}};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  size_t m = in.size();
  int n = 8;
  Dijkstra::graph_t graph(n);
  for (size_t i = 0; i < m; i++) {
    int u = std::get<0>(in[i]);
    int v = std::get<1>(in[i]);
    int w = std::get<2>(in[i]);

    graph.add_edge(u, Dijkstra::edge_t(v, w));
    graph.add_edge(v, Dijkstra::edge_t(u, w));
  };

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  Dijkstra::MyDijkstra testTaskSequential(taskDataSeq);
  testTaskSequential.graph = std::move(graph);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(testTaskSequential.out, out);
}
