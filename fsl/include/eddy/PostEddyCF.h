// Declarations of classes and functions that
// perform a post-hoc registration of the shells
// for the eddy project/.
//
// post_registration.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford
//

#ifndef PostEddyCF_h
#define PostEddyCF_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include "armawrap/newmat.h"
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS           // To allow us to use .set_sform etc
#endif
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "miscmaths/nonlin.h"
#include "EddyHelperClasses.h"
#include "ECModels.h"

namespace EDDY {


/****************************************************************//**
*
* \brief Class used to implement a Mutual Information cost-
* function for post-hoc registration of shells in the eddy project.
*
* Class used to implement a Mutual Information cost-
* function for post-hoc registration of shells in the eddy project.
* It is implemented using the "Pimpl idiom" which means that this class
* only implements an interface whereas the actual work is being performed
* by the PostEddyCFImpl class which is declared and defined in
* PostEddyCF.cpp or cuda/PostEddyCF.cu depending on what platform
* the code is compiled for.
*
********************************************************************/
class PostEddyCFImpl;
class PostEddyCF : public MISCMATHS::NonlinCF
{
public:
  PostEddyCF(const NEWIMAGE::volume<float>&  ref,
	     const NEWIMAGE::volume<float>&  ima,
	     const NEWIMAGE::volume<float>&  mask,
	     unsigned int                    nbins);
  PostEddyCF(const NEWIMAGE::volume<float>&  ref,
	     const NEWIMAGE::volume<float>&  ima,
	     const NEWIMAGE::volume<float>&  mask,
	     unsigned int                    nbins,
	     unsigned int                    pe_dir);
  ~PostEddyCF();
  NEWIMAGE::volume<float> GetTransformedIma(const NEWMAT::ColumnVector& p) const;
  double cf(const NEWMAT::ColumnVector& p) const;
  NEWMAT::ReturnMatrix grad(const NEWMAT::ColumnVector& p) const;
private:
  int                        _pe_dir;  // 0 for x, 1 for y
  MutualInfoHelper           _fwd_mih;
  MutualInfoHelper           _bwd_mih;
  PostEddyCFImpl             *_pimpl;
};

} // End namespace EDDY

#endif // End #ifndef PostEddyCF_h
