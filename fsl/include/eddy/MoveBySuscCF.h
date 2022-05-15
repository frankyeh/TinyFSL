// Declarations of classes and functions that
// calculates the derivative fields for the
// movement-by-susceptibility modelling.
//
// MoveBySuscCF.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2016 University of Oxford
//

#ifndef MoveBySuscCF_h
#define MoveBySuscCF_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>

#include "armawrap/newmat.h"
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS           // To allow us to use .set_sform etc
#endif
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "miscmaths/nonlin.h"
#include "EddyHelperClasses.h"
#include "ECModels.h"
#include "EddyCommandLineOptions.h"

namespace EDDY {


/****************************************************************//**
*
* \brief Class used to implement a cost-function for estimating
* the derivative fields of a movement-by-susceptibility model
* in the eddy project.
*
* Class used to implement a Mutual Information cost-
* function for post-hoc registration of shells in the eddy project.
* It is implemented using the "Pimpl idiom" which means that this class
* only implements an interface whereas the actual work is being performed
* by the MoveBySuscCFImpl class which is declared and defined in
* MoveBySuscCF.cpp or cuda/MoveBySuscCF.cu depending on what platform
* the code is compiled for.
*
********************************************************************/
class MoveBySuscCFImpl;
class MoveBySuscCF : public MISCMATHS::NonlinCF
{
public:
  MoveBySuscCF(EDDY::ECScanManager&                 sm,
	       const EDDY::EddyCommandLineOptions&  clo,
	       const std::vector<unsigned int>&     b0s,
	       const std::vector<unsigned int>&     dwis,
	       const std::vector<unsigned int>&     mps,
	       unsigned int                         order,
	       double                               ksp);
  ~MoveBySuscCF();
  double cf(const NEWMAT::ColumnVector& p) const;
  NEWMAT::ReturnMatrix grad(const NEWMAT::ColumnVector& p) const;
  std::shared_ptr<MISCMATHS::BFMatrix> hess(const NEWMAT::ColumnVector& p,
				   std::shared_ptr<MISCMATHS::BFMatrix> iptr=std::shared_ptr<MISCMATHS::BFMatrix>()) const;
  void SetLambda(double lambda);
  NEWMAT::ReturnMatrix Par() const;
  unsigned int NPar() const;
  void WriteFirstOrderFields(const std::string& fname) const;
  void WriteSecondOrderFields(const std::string& fname) const;
  void ResetCache();
private:
  mutable MoveBySuscCFImpl             *_pimpl;
};

} // End namespace EDDY

#endif // End #ifndef MoveBySuscCF_h
