// Copyright 2024 Savotina Valeria
#include "seq/savotina_v_grahams_alg/include/point.hpp"

SavotinaPoint::SavotinaPoint() {
  x = 0;
  y = 0;
}

SavotinaPoint::SavotinaPoint(double X, double Y) {
  x = X;
  y = Y;
}

SavotinaPoint::SavotinaPoint(const SavotinaPoint& p2) {
  x = p2.x;
  y = p2.y;
}

SavotinaPoint& SavotinaPoint::operator=(const SavotinaPoint& p2) {
  if (this == &p2) return *this;
  x = p2.x;
  y = p2.y;
  return *this;
}

bool SavotinaPoint::operator==(const SavotinaPoint& p2) const {
  bool res = false;
  if (x == p2.x && y == p2.y) res = true;
  return res;
}

double SavotinaPoint::Distance(const SavotinaPoint& p) const {
  return sqrt((p.x - x) * (p.x - x) + (p.y - y) * (p.y - y));
}

void SavotinaPoint::Replace(SavotinaPoint& p2) {
  SavotinaPoint tmp = (*this);
  (*this) = p2;
  p2 = tmp;
}

int SavotinaPoint::Compare(const SavotinaPoint& pivot, const SavotinaPoint& P) const {
  int res = 0;  // ==
  double x1 = pivot.x - x;
  double y1 = pivot.y - y;
  double x2 = P.x - pivot.x;
  double y2 = P.y - pivot.y;
  double val = x1 * y2 - y1 * x2;

  if (val > 0)
    res = 1;  // 2 upper 1 | 2 > 1
  else if (val < 0)
    res = -1;  // 2 downer 1 | 2 < 1
  else {
    float dist1 = (*this).Distance(pivot);
    float dist2 = (*this).Distance(P);

    if (dist1 != 0.0 || dist2 != 0.0) {
      if (dist2 > dist1)
        res = 2;  // 2 > 1
      else if (dist2 < dist1)
        res = -2;  // 2 < 1
    }
  }

  return res;
}

SavotinaPoint SavotinaPoint::aRandomPoint(double min, double max) {
  std::random_device rd;
  std::default_random_engine gen{rd()};
  std::uniform_real_distribution<double> random(min, max);

  return SavotinaPoint(random(gen), random(gen));
}
