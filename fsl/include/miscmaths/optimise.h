/*  optimise.h

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 1999-2000 University of Oxford  */

/*  CCOPYRIGHT  */

// Mathematical optimisation functions


#if !defined(__optimise_h)
#define __optimise_h

#include <cmath>
#include "armawrap/newmatap.h"
#include "string"

namespace MISCMATHS {

  float optimise1d(NEWMAT::ColumnVector &pt, const NEWMAT::ColumnVector dir,
		const NEWMAT::ColumnVector tol, int &iterations_done,
		float (*func)(const NEWMAT::ColumnVector &), int max_iter,
		float &init_value, float boundguess);


 float optimise(NEWMAT::ColumnVector &pt, int numopt, const NEWMAT::ColumnVector &tol,
		float (*func)(const NEWMAT::ColumnVector &), int &iterations_done,
		int max_iter, const NEWMAT::ColumnVector& boundguess,
		const std::string type="brent");

}

#endif
