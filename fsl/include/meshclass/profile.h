/*  Copyright (C) 1999-2004 University of Oxford  */

/*  CCOPYRIGHT */

#ifndef _profile
#define _profile

#include <iostream>
#include <vector>
#include <cstdlib>


struct pro_pair
{
  double abs;
  double val;
};

class Profile
{
 private:
  int lroi, rroi;
  bool mindef, maxdef;
  int amin, amax;


 public:

  std::vector<pro_pair > v;

  Profile();
  ~Profile();

  int size() const;
  void print() const;
  void add (const double, const double);
  void init_roi();
  void set_lroi(const double);
  void set_rroi(const double);
  const double value(const double) const;
  const double min();
  const double max();
  const double minabs();
  const double maxabs();
  const double threshold(const double d);
  const double begin();
  const double end();
  const double next_point_over (const double, const double);
  const double next_point_under (const double, const double);
  const double last_point_over (const double, const double);
  const double last_point_under (const double, const double);

};

#endif
