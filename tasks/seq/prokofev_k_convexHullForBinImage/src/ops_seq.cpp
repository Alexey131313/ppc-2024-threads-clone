// Copyright 2024 Prokofev Kirill
#include "seq/prokofev_k_convexHullForBinImage/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool BinaryImageConvexHullSeq::pre_processing() {
  internal_order_test();
  width = taskData->inputs_count[0];
  height = taskData->inputs_count[1];
  img = reinterpret_cast<decltype(img)>(taskData->inputs[0]);
  img_ = *img;
  res = {};
  return true;
}

bool BinaryImageConvexHullSeq::validation() {
  internal_order_test();
  return taskData->inputs_count.size() == 2 && taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool BinaryImageConvexHullSeq::run() {
  internal_order_test();
  std::vector<int> local_image = FindComponents(img_, width, height);
  int count_components = FindCountComponents(local_image);
  for (int i = 1; i <= count_components; ++i) {
    std::vector<int> points = RemoveExtraPoints(local_image, width, height, i);
    std::vector<int> ch = GrahamAlgorithm(points);
    for (size_t j = 0; j < ch.size(); ++j) {
      res.emplace_back(ch[j]);
    }
    res.emplace_back(-1);
  }

  return true;
}

bool BinaryImageConvexHullSeq::post_processing() {
  internal_order_test();
  std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}

void floodFill(std::vector<int>* image, int height, int width, int yStart, int xStart, int label) {
  std::queue<Point> tasks;
  tasks.emplace(xStart, yStart);
  while (!tasks.empty()) {
    int x = tasks.front().x;
    int y = tasks.front().y;
    tasks.pop();
    if (x >= 0 && y >= 0 && y < height && x < width && image->at(y * width + x) == 1) {
      (*image)[y * width + x] = label;
      tasks.emplace(x - 1, y - 1);
      tasks.emplace(x - 1, y);
      tasks.emplace(x - 1, y + 1);
      tasks.emplace(x, y + 1);
      tasks.emplace(x + 1, y + 1);
      tasks.emplace(x + 1, y);
      tasks.emplace(x, y - 1);
      tasks.emplace(x + 1, y - 1);
    }
  }
}

std::vector<int> FindComponents(const std::vector<std::vector<int>>& image, int width, int height) {
  std::vector<int> image_with_components(width * height);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      image_with_components[i * width + j] = image[i][j];
    }
  }
  int label = 2;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      if (image_with_components[i * width + j] == 1) {
        floodFill(&image_with_components, height, width, i, j, label);
        ++label;
      }
    }
  }
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      if (image_with_components[i * width + j] != 0) {
        image_with_components[i * width + j]--;
      }
    }
  }
  return image_with_components;
}

int FindCountComponents(const std::vector<int>& image) {
  int count_components = 0;
  for (size_t i = 0; i < image.size(); ++i) {
    if (image[i] > count_components) {
      count_components = image[i];
    }
  }
  return count_components;
}

int FindCountPointsInComponent(const std::vector<int>& image) {
  int count_points = 0;
  for (size_t i = 0; i < image.size(); ++i) {
    if (image[i] != 0) {
      count_points++;
    }
  }
  return count_points;
}

std::vector<int> RemoveExtraPoints(const std::vector<int>& image, int width, int height, int label) {
  std::vector<int> local_image(image);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      if (image[i * width + j] == label) {
        if ((j > 0) && (j < width - 1)) {
          if ((i == 0) || (i == height - 1)) {
            if ((image[i * width + j - 1] == label) && (image[i * width + j + 1] == label)) {
              local_image[i * width + j] = 0;
            }
          }
          if ((i > 0) && (i < height - 1)) {
            if (((image[i * width + j - 1] == label) && (image[i * width + j + 1] == label)) ||
                ((image[(i + 1) * width + j] == label) && (image[(i - 1) * width + j] == label))) {
              local_image[i * width + j] = 0;
            }
          }
          continue;
        }
        if ((i > 0) && (i < height - 1)) {
          if ((j == 0) || (j == width - 1)) {
            if ((image[(i - 1) * width + j] == label) && (image[(i + 1) * width + j] == label)) {
              local_image[i * width + j] = 0;
            }
          }
          if ((j > 0) && (j < width - 1)) {
            if (((image[i * width + j - 1] == label) && (image[i * width + j + 1] == label)) ||
                ((image[(i + 1) * width + j] == label) && (image[(i - 1) * width + j] == label))) {
              local_image[i * width + j] = 0;
            }
          }
        }
      } else {
        local_image[i * width + j] = 0;
      }
    }
  }
  int size = FindCountPointsInComponent(local_image);
  std::vector<int> points(size * 2);
  int k = 0;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      if (local_image[i * width + j] != 0) {
        points[k] = j;
        k++;
        points[k] = i;
        k++;
      }
    }
  }
  return points;
}

int Cross(int x1, int y1, int x2, int y2, int x3, int y3) { return ((x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1)); }

void Sort(std::vector<int>* points, int xMin, int yMin) {
  int size = points->size() / 2;
  for (int i = 1; i < size; ++i) {
    int j = i;
    while ((j > 0) && (Cross(xMin, yMin, (*points)[2 * j - 2], (*points)[2 * j - 1], (*points)[2 * j],
                             (*points)[2 * j + 1]) < 0)) {
      int temp = (*points)[2 * j - 2];
      (*points)[2 * j - 2] = (*points)[2 * j];
      (*points)[2 * j] = temp;
      temp = (*points)[2 * j - 1];
      (*points)[2 * j - 1] = (*points)[2 * j + 1];
      (*points)[2 * j + 1] = temp;
      j--;
    }
  }
}

std::vector<int> GrahamAlgorithm(std::vector<int> points) {
  std::vector<int> result;
  int num_points = points.size() / 2;
  if (num_points > 1) {
    int x_min = points[0];
    int y_min = points[1];
    int min_index = 0;
    for (size_t i = 2; i < points.size(); i += 2) {
      if (points[i] < x_min || (points[i] == x_min && points[i + 1] < y_min)) {
        x_min = points[i];
        y_min = points[i + 1];
        min_index = i;
      }
    }
    int temp = points[min_index];
    points[min_index] = points[num_points * 2 - 2];
    points[num_points * 2 - 2] = temp;
    temp = points[min_index + 1];
    points[min_index + 1] = points[num_points * 2 - 1];
    points[num_points * 2 - 1] = temp;
    points.pop_back();
    points.pop_back();
    Sort(&points, x_min, y_min);
    result.emplace_back(x_min);
    result.emplace_back(y_min);
    result.emplace_back(points[0]);
    result.emplace_back(points[1]);
    for (size_t i = 2; i < points.size(); i += 2) {
      int result_size = result.size();
      int x1 = result[result_size - 4];
      int y1 = result[result_size - 3];
      int x2 = result[result_size - 2];
      int y2 = result[result_size - 1];
      int x3 = points[i];
      int y3 = points[i + 1];

      int rot = Cross(x1, y1, x2, y2, x3, y3);
      if (rot == 0) {
        result[result_size - 2] = x3;
        result[result_size - 1] = y3;
      } else if (rot < 0) {
        while (Cross(result[(result.size()) - 4], result[(result.size()) - 3], result[(result.size()) - 2],
                     result[(result.size()) - 1], x3, y3) < 0)
          result.pop_back(), result.pop_back();
        result.emplace_back(x3);
        result.emplace_back(y3);
      } else {
        result.emplace_back(x3);
        result.emplace_back(y3);
      }
    }
  } else {
    result.resize(2);
    result[0] = points[0];
    result[1] = points[1];
  }
  return result;
}
