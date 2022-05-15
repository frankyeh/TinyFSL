/*  minimize

    Tim Behrens, FMRIB Image Analysis Group

    Copyright (C) 1999-2000 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(minimize_h)
#define minimize_h

#include <string>
#include <iostream>
#include <fstream>
//#include <unistd.h>
#include <vector>
#include <algorithm>
#include "armawrap/newmatap.h"
#include "armawrap/newmatio.h"
#include "miscmaths.h"


///////////////////////////////////////////////////////

//fminsearch.m

namespace MISCMATHS {

class pair_comparer
{
public:
  bool operator()(const std::pair<float,NEWMAT::ColumnVector>& p1,const std::pair<float,NEWMAT::ColumnVector>& p2) const
  {
    return p1.first < p2.first;
  }
};

  class EvalFunction;
  class gEvalFunction;

float diff1(const NEWMAT::ColumnVector& x, const EvalFunction& func, int i,float h,int errorord=4);// finite diff derivative

float diff2(const NEWMAT::ColumnVector& x, const EvalFunction& func, int i,float h,int errorord=4);// finite diff 2nd derivative

float diff2(const NEWMAT::ColumnVector& x, const EvalFunction& func, int i,int j,float h,int errorord=4);// finite diff cross derivative

NEWMAT::ReturnMatrix gradient(const NEWMAT::ColumnVector& x, const EvalFunction& func,float h,int errorord=4);// finite diff derivative vector

NEWMAT::ReturnMatrix hessian(const NEWMAT::ColumnVector& x, const EvalFunction& func,float h,int errorord=4);// finite diff hessian

void minsearch(NEWMAT::ColumnVector& x, const EvalFunction& func, NEWMAT::ColumnVector& paramstovary);

void scg(NEWMAT::ColumnVector& x, const gEvalFunction& func, NEWMAT::ColumnVector& paramstovary, float tol = 0.0000001, float eps=1e-16, int niters=500);

class EvalFunction
{//Function where gradient is not analytic (or you are too lazy to work it out) (required for fminsearch)
public:
  EvalFunction(){}
  virtual float evaluate(const NEWMAT::ColumnVector& x) const = 0; //evaluate the function
  virtual ~EvalFunction(){};

  virtual void minimize(NEWMAT::ColumnVector& x)
  {
    NEWMAT::ColumnVector paramstovary(x.Nrows());
    paramstovary = 1;
    minsearch(x,*this,paramstovary);
  }

  virtual void minimize(NEWMAT::ColumnVector& x, NEWMAT::ColumnVector& paramstovary)
  {
    minsearch(x,*this,paramstovary);
  }

private:
  const EvalFunction& operator=(EvalFunction& par);
  EvalFunction(const EvalFunction&);
};

class gEvalFunction : public EvalFunction
{//Function where gradient is analytic (required for scg)
public:
  gEvalFunction() : EvalFunction(){}
  // evaluate is inherited from EvalFunction

  virtual NEWMAT::ReturnMatrix g_evaluate(const NEWMAT::ColumnVector& x) const = 0; //evaluate the gradient
  virtual ~gEvalFunction(){};

  virtual void minimize(NEWMAT::ColumnVector& x)
  {
    NEWMAT::ColumnVector paramstovary(x.Nrows());
    paramstovary = 1;
    scg(x,*this,paramstovary);
  }

  virtual void minimize(NEWMAT::ColumnVector& x, NEWMAT::ColumnVector& paramstovary)
  {
    scg(x,*this,paramstovary);
  }

private:

  const gEvalFunction& operator=(gEvalFunction& par);
  gEvalFunction(const gEvalFunction&);
};

}


#endif
