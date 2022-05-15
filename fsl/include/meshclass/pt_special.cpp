/*  Copyright (C) 1999-2004 University of Oxford  */

/*  CCOPYRIGHT */

#include "pt_special.h"
#include "mesh.h"
#include "mpoint.h"
#include "triangle.h"

namespace mesh{

  Pt_special::Pt_special()
  {
    P = NULL;
    T.clear();
  }

  const bool Pt_special::operator<(const Pt_special & P2)
  {
    return (this->P->get_coord().X < P2.P->get_coord().X);
  }
  
  const bool compPt::operator()(Pt_special const* p1, Pt_special const* p2) const
  {
    return (p1->P->get_coord().X < p2->P->get_coord().X);
  }
  
}



