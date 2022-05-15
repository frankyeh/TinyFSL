/*  rungekutta.h

    Mark Woolrich - FMRIB Image Analysis Group

    Copyright (C) 2002 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(rungekutta_h)
#define rungekutta_h

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "armawrap/newmatap.h"
#include "armawrap/newmatio.h"

namespace MISCMATHS {

class Derivative
{
public:
  Derivative(int pny) : ny(pny), dy(pny) {}

  // x is time point to evaluate at
  // y is state variables
  // paramvalues are "constants" in the diff eqn
  virtual const NEWMAT::ColumnVector& evaluate(float x,const NEWMAT::ColumnVector& y,const NEWMAT::ColumnVector& paramvalues) const = 0;

  virtual ~Derivative(){};

protected:
  int ny;
  mutable NEWMAT::ColumnVector dy;
};

void rk(NEWMAT::ColumnVector& ret, const NEWMAT::ColumnVector& y, const NEWMAT::ColumnVector& dy, float x, float h, const Derivative& deriv,const NEWMAT::ColumnVector& paramvalues);

void rkqc(NEWMAT::ColumnVector& y, float& x, float& hnext, NEWMAT::ColumnVector& dy, float htry, float eps, const Derivative& deriv,const NEWMAT::ColumnVector& paramvalues);

void runge_kutta(NEWMAT::Matrix& yp, NEWMAT::ColumnVector& xp, NEWMAT::ColumnVector& hp, const NEWMAT::ColumnVector& ystart, float x1, float x2, float eps, float hmin, const Derivative& deriv,const NEWMAT::ColumnVector& paramvalues);

}

#endif
