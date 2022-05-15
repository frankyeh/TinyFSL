/*  miscprob.h

    Christian Beckmann & Mark Woolrich, FMRIB Image Analysis Group

    Copyright (C) 1999-2000 University of Oxford  */

/*  CCOPYRIGHT  */

// Miscellaneous maths functions that rely on libprob build ontop of miscmaths


#if !defined(__miscprob_h)
#define __miscprob_h

#include "miscmaths.h"
#include "cprob/libprob.h"
#include "stdlib.h"

namespace MISCMATHS {

//   ReturnMatrix betarnd(const int dim1, const int dim2,
// 		       const float a, const float b);

  NEWMAT::ReturnMatrix betapdf(const NEWMAT::RowVector& vals,
		       const float a, const float b);

  NEWMAT::ReturnMatrix unifrnd(const int dim1 = 1, const int dim2 = -1,
		       const float start = 0, const float end = 1);

  NEWMAT::ReturnMatrix normrnd(const int dim1 = 1, const int dim2 = -1,
		       const float mu = 0, const float sigma = 1);

  // returns nsamps*nparams matrix:
  NEWMAT::ReturnMatrix mvnrnd(const NEWMAT::RowVector& mu, const NEWMAT::SymmetricMatrix& covar, int nsamp = 1);

  float mvnpdf(const NEWMAT::RowVector& vals, const NEWMAT::RowVector& mu, const NEWMAT::SymmetricMatrix& covar);

  float bvnpdf(const NEWMAT::RowVector& vals, const NEWMAT::RowVector& mu, const NEWMAT::SymmetricMatrix& covar);

  float normpdf(const float val, const float mu = 0, const float var = 1);
  float lognormpdf(const float val, const float mu = 0, const float var = 1);

  NEWMAT::ReturnMatrix normpdf(const NEWMAT::RowVector& vals, const float mu = 0, const float var = 1);

  NEWMAT::ReturnMatrix normpdf(const NEWMAT::RowVector& vals, const NEWMAT::RowVector& mus,
		       const NEWMAT::RowVector& vars);

  NEWMAT::ReturnMatrix normcdf(const NEWMAT::RowVector& vals, const float mu = 0, const float var = 1);

  NEWMAT::ReturnMatrix gammapdf(const NEWMAT::RowVector& vals, const float mu = 0, const float var = 1);

  NEWMAT::ReturnMatrix gammacdf(const NEWMAT::RowVector& vals, const float mu = 0, const float var = 1);

//   NEWMAT::ReturnMatrix gammarnd(const int dim1, const int dim2,
// 			const float a, const float b);

  // returns n! * n matrix of all possible permutations
  NEWMAT::ReturnMatrix perms(const int n);


  class Mvnormrandm
    {
    public:
      Mvnormrandm(){}

      Mvnormrandm(const NEWMAT::RowVector& pmu, const NEWMAT::SymmetricMatrix& pcovar) :
	mu(pmu),
	covar(pcovar)
	{
	  NEWMAT::Matrix eig_vec;
	  NEWMAT::DiagonalMatrix eig_val;
	  EigenValues(covar,eig_val,eig_vec);

	  covarw = sqrt(eig_val)*eig_vec.t();
	}

      NEWMAT::ReturnMatrix next(int nsamp = 1) const
	{
	  NEWMAT::Matrix ret = ones(nsamp, 1)*mu + normrnd(nsamp,mu.Ncols())*covarw;
	  ret.Release();
	  return ret;
	}

      NEWMAT::ReturnMatrix next(const NEWMAT::RowVector& pmu, int nsamp = 1)
	{
	  mu=pmu;

	  NEWMAT::Matrix ret = ones(nsamp, 1)*mu + normrnd(nsamp,mu.Ncols())*covarw;
	  ret.Release();
	  return ret;
	}

      void setcovar(const NEWMAT::SymmetricMatrix& pcovar)
	{
	  covar=pcovar;

	  mu.ReSize(covar.Nrows());
	  mu=0;

	  NEWMAT::Matrix eig_vec;
	  NEWMAT::DiagonalMatrix eig_val;
	  EigenValues(covar,eig_val,eig_vec);

	  covarw = sqrt(eig_val)*eig_vec.t();
	}

    private:

      NEWMAT::RowVector mu;
      NEWMAT::SymmetricMatrix covar;

      NEWMAT::Matrix covarw;

    };
}
#endif
