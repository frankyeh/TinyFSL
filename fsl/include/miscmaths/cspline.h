/*  cspline

    Cubic spline fitting and interpolation
    Tim Behrens, FMRIB Image Analysis Group

    Copyright (C) 1999-2000 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(__cspline_h)
#define __cspline_h

#include <string>
#include <iostream>
#include <fstream>

#include "armawrap/newmatap.h"
#include "armawrap/newmatio.h"
#include "miscmaths.h"

#define WANT_STREAM
#define WANT_MATH


///////////////////////////////////////////////////////


namespace MISCMATHS {

  class Cspline{
  public:
    Cspline(){}
    Cspline(NEWMAT::ColumnVector& pnodes,NEWMAT::ColumnVector& pvals):
      nodes(pnodes),
      vals(pvals),
      n(nodes.Nrows())
    {
      fit();
      fitted=true;
    }

    Cspline(NEWMAT::ColumnVector& pnodes, NEWMAT::Matrix& pcoefs) :
      nodes(pnodes),
      coefs(pcoefs),
      n(nodes.Nrows())
    { fitted=true;}

    ~Cspline(){
      fitted=false;
    };

    void set(NEWMAT::ColumnVector& pnodes,NEWMAT::ColumnVector& pvals);
    void set(NEWMAT::ColumnVector& pnodes, NEWMAT::Matrix& pcoefs);

    void fit();
    float interpolate(float xx) const;
    float interpolate(float xx,int ind) const;
    NEWMAT::ColumnVector interpolate(const NEWMAT::ColumnVector& x) const;
    NEWMAT::ColumnVector interpolate(const NEWMAT::ColumnVector& x, const NEWMAT::ColumnVector& indvec) const;

  protected:

    bool fitted;
    NEWMAT::ColumnVector nodes;
    NEWMAT::ColumnVector vals;
    NEWMAT::Matrix coefs;
    int n;
    void diff(const NEWMAT::ColumnVector& x, NEWMAT::ColumnVector& dx );

  };
}


#endif
