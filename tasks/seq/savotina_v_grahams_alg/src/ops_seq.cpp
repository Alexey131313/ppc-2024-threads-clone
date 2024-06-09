// Copyright 2024 Savotina Valeria
#include "seq/savotina_v_grahams_alg/include/ops_seq.hpp"

// Quick Sort
void SavotinaQuickSort(std::vector<SavotinaPoint>& pointArr, int left, int right) {
  if (left > right) return;  // Leave recursion

  int piv = 0;
  int arrSize = right - left + 1;
  int L = left;
  int R = right;

  // Index of pivot
  if ((arrSize % 2) == 0)
    piv = (arrSize / 2 - 1) + left;
  else
    piv = std::trunc(arrSize / 2) + left;

  SavotinaPoint pivot = pointArr[piv];

  while (L <= R) {
    while (!(pointArr[0].Compare(pivot, pointArr[L]) >= 0)) ++L;

    while (!(pointArr[0].Compare(pivot, pointArr[R]) <= 0)) --R;

    if (L <= R) {
      pointArr[L].Replace(pointArr[R]);
      ++L;
      --R;
    }
  }

  SavotinaQuickSort(pointArr, left, R);
  SavotinaQuickSort(pointArr, L, right);
}

// A search minimum point in point's array (min x, then min y)
SavotinaPoint SavotinaMinPoint(const std::vector<SavotinaPoint>& pointArr) {
  double minX = pointArr[0].x;
  double minY = 0;
  std::stack<int> S;
  S.push(0);
  int pArrSize = pointArr.size();

  // Search min x
  for (int i = 1; i < pArrSize; ++i) {
    double Xi = pointArr[i].x;
    if (Xi < minX) {
      minX = Xi;
      while (!S.empty()) S.pop();
      S.push(i);
    } else if (Xi == minX) {
      S.push(i);
    }
  }

  // Search min y
  minY = pointArr[S.top()].y;
  S.pop();
  while (!S.empty()) {
    double Yi = pointArr[S.top()].y;
    if (Yi < minY) minY = Yi;
    S.pop();
  }

  return SavotinaPoint(minX, minY);  // Anonymous object
}

// Determining the number (index) of a point in the array
int SavotinaPointPosition(const SavotinaPoint& p, const std::vector<SavotinaPoint>& pointArr) {
  int pp = 0;
  int pArrSize = pointArr.size();
  for (int i = 0; i < pArrSize; ++i) {
    if (pointArr[i].x == p.x && pointArr[i].y == p.y) {
      pp = i;
      break;
    }
  }
  return pp;
}

// Creating a minimum convex hull
std::vector<SavotinaPoint> SavotinaMinConvexHull(std::vector<SavotinaPoint> pointArr) {
  if (pointArr.size() < 3) return pointArr;

  std::vector<SavotinaPoint> mch;
  int ind1 = 0;
  int ind2 = 1;
  int arrSize = pointArr.size();

  while (pointArr[ind1] == pointArr[ind2]) {
    ++ind2;
    if (ind2 == arrSize) break;
  }

  mch.push_back(pointArr[ind1]);
  if (ind2 != arrSize) {
    mch.push_back(pointArr[ind2]);

    arrSize = pointArr.size();
    int val = 0;
    int mchSize = 0;

    for (int i = ind2 + 1; i < arrSize; ++i) {
      mchSize = mch.size();
      val = mch[mchSize - 2].Compare(mch[mchSize - 1], pointArr[i]);

      if ((val == -1) || (val == 2))  // (val == -1) 2 < 1 || (val == 2) dist2 > dist1, 2 > 1
      {
        mch.erase(mch.end() - 1);
        if (mch.size() < 2) mch.push_back(pointArr[i]);
        --i;
      } else if (val == 1)  // (val == 1)  2 > 1
      {
        mch.push_back(pointArr[i]);
      }
    }
  }

  return mch;
}

std::vector<SavotinaPoint> SavotinaRandomPoints(double leftBorder, double rightBorder, size_t size) {
  std::vector<SavotinaPoint> arrPoints(size);
  for (SavotinaPoint& value : arrPoints) value = SavotinaPoint::aRandomPoint(leftBorder, rightBorder);
  return arrPoints;
}

bool SavotinaGrahamsAlgorithmSequential::pre_processing() {
  internal_order_test();

  // Init value for input and output
  pointsArr = std::vector<SavotinaPoint>(taskData->inputs_count[0]);

  auto* pArray = reinterpret_cast<SavotinaPoint*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    pointsArr[i] = pArray[i];
  }

  minConvexHull = pointsArr;
  return true;
}

bool SavotinaGrahamsAlgorithmSequential::validation() {
  internal_order_test();

  // Check count elements of output
  return (taskData->outputs_count[0] <= taskData->inputs_count[0]);
}

bool SavotinaGrahamsAlgorithmSequential::run() {
  internal_order_test();

  if (pointsArr.empty()) return true;

  // Step 1: search the minimum point P0
  SavotinaPoint P0 = SavotinaMinPoint(pointsArr);
  int p0 = SavotinaPointPosition(P0, pointsArr);

  // Step 2: sort all points except P0
  minConvexHull[0].Replace(minConvexHull[p0]);
  SavotinaQuickSort(minConvexHull, 1, minConvexHull.size() - 1);  // Quick Sort

  // Step 3: build a minimum convex hull
  minConvexHull = SavotinaMinConvexHull(minConvexHull);
  return true;
}

bool SavotinaGrahamsAlgorithmSequential::post_processing() {
  internal_order_test();

  std::copy(minConvexHull.begin(), minConvexHull.end(), reinterpret_cast<SavotinaPoint*>(taskData->outputs[0]));
  return true;
}
