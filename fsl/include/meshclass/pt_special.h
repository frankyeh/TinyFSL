/*  Copyright (C) 1999-2004 University of Oxford  */

/*  CCOPYRIGHT */

#ifndef _ptspecial
#define _ptspecial

#include <list>

namespace mesh {

class Triangle;
class Mpoint;

  //this class is only used for the real_self_intersection algorithm

class Pt_special
{
 public:
  Mpoint * P;
  std::list<Triangle *> T;
  Pt_special();
  const bool operator<(const Pt_special & P2);
};

struct compPt
{
  const bool operator()(Pt_special const* p1, Pt_special const* p2) const;
};

}

#endif
