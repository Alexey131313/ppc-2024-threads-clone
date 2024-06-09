// Copyright 2024 Savotina Valeria
#pragma once

#include <cmath>
#include <random>

struct SavotinaPoint {
  double x;
  double y;

  SavotinaPoint();
  SavotinaPoint(double X, double Y);
  SavotinaPoint(const SavotinaPoint& p2);

  SavotinaPoint& operator=(const SavotinaPoint& p2);
  bool operator==(const SavotinaPoint& p2) const;

  double Distance(const SavotinaPoint& p) const;
  void Replace(SavotinaPoint& p2);
  int Compare(const SavotinaPoint& pivot, const SavotinaPoint& P) const;
  static SavotinaPoint aRandomPoint(double min, double max);
};
