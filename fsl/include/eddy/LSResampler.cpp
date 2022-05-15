/*! \file LSResampler.cpp
    \brief Contains definition of CPU implementation of a class for least-squares resampling of pairs of images

    \author Jesper Andersson
    \version 1.0b, August, 2013.
*/
//
// LSResampler.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford
//

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <time.h>
#include "armawrap/newmat.h"
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS           // To allow us to use .set_sform etc
#endif
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "warpfns/warpfns.h"
#include "topup/topup_file_io.h"
#include "topup/displacement_vector.h"
#include "EddyHelperClasses.h"
#include "EddyUtils.h"
#include "ECScanClasses.h"
#include "LSResampler.h"

namespace EDDY {

class LSResamplerImpl
{
public:
  LSResamplerImpl(const EDDY::ECScan&                             s1,
		  const EDDY::ECScan&                             s2,
		  std::shared_ptr<const NEWIMAGE::volume<float> > hzfield,
		  double                                          lambda);
  const NEWIMAGE::volume<float>& GetResampledVolume() const EddyTry { return(_rvol); } EddyCatch
  const NEWIMAGE::volume<float>& GetMask() const EddyTry { return(_omask); } EddyCatch
private:
  NEWIMAGE::volume<float> _rvol;  // Resampled volume
  NEWIMAGE::volume<float> _omask; // Mask indicating valid voxels in _rvol
  template<typename T>
  T sqr(const T& v) const { return(v*v); }
};

LSResampler::LSResampler(const EDDY::ECScan&                             s1,
			 const EDDY::ECScan&                             s2,
			 std::shared_ptr<const NEWIMAGE::volume<float> > hzfield,
			 double                                          lambda) EddyTry
{
  _pimpl = new LSResamplerImpl(s1,s2,hzfield,lambda);
} EddyCatch

LSResampler::~LSResampler()
{
  delete _pimpl;
}

const NEWIMAGE::volume<float>& LSResampler::GetResampledVolume() const EddyTry
{
  return(_pimpl->GetResampledVolume());
} EddyCatch

const NEWIMAGE::volume<float>& LSResampler::GetMask() const EddyTry
{
  return(_pimpl->GetMask());
} EddyCatch

/****************************************************************//**
*
* Constructs an LSResamplerImpl object.
* All the work for resampling a pair of scans into a single volume
* is carried out inside this constructor. After the object has been
* constructed one can immediately obtain the resampled volume through
* a call to GetResampledVolume.
* \param s1 One of a pair of scans with parallel phase-encodings
* \param s2 The second of a pair of scans with parallel phase-encodings
* \param hzfield Field in Hz in model space
*
********************************************************************/

LSResamplerImpl::LSResamplerImpl(const EDDY::ECScan&                             s1,
				 const EDDY::ECScan&                             s2,
				 std::shared_ptr<const NEWIMAGE::volume<float> > hzfield,
				 double                                          lambda) EddyTry
{
  // cout << "I'm in CPU version of LSResampler" << endl;
  if (!EddyUtils::AreMatchingPair(s1,s2)) throw EddyException("LSResampler::LSResampler:: Mismatched pair");
  // Resample both images using rigid body parameters
  NEWIMAGE::volume<float> mask1, mask2;
  NEWIMAGE::volume<float> ima1 = s1.GetMotionCorrectedIma(mask1);
  NEWIMAGE::volume<float> ima2 = s2.GetMotionCorrectedIma(mask2);
  NEWIMAGE::volume<float> mask = mask1*mask2;
  _omask.reinitialize(mask.xsize(),mask.ysize(),mask.zsize());
  _omask = 1.0;
  _rvol.reinitialize(ima1.xsize(),ima1.ysize(),ima1.zsize());
  NEWIMAGE::copybasicproperties(ima1,_rvol);
  // Get fields
  NEWIMAGE::volume4D<float> field1 = s1.FieldForScanToModelTransform(hzfield); // In mm
  NEWIMAGE::volume4D<float> field2 = s2.FieldForScanToModelTransform(hzfield); // In mm
  // Check what direction phase-encode is in
  bool pex = false;
  unsigned int sz = field1[0].ysize();
  if (s1.GetAcqPara().PhaseEncodeVector()(1)) { pex = true; sz = field1[0].xsize(); }
  TOPUP::DispVec dv1(sz), dv2(sz);
  NEWMAT::Matrix StS = dv1.GetS_Matrix(false);
  StS = lambda*StS.t()*StS;
  NEWMAT::ColumnVector zeros;
  if (pex) zeros.ReSize(ima1.xsize()); else zeros.ReSize(ima1.ysize());
  zeros = 0.0;
  double sf1, sf2; // Scale factors mm->voxels
  if (pex) { sf1 = 1.0/ima1.xdim(); sf2 = 1.0/ima2.xdim(); }
  else { sf1 = 1.0/ima1.ydim(); sf2 = 1.0/ima2.ydim(); }
  // For all slices
  for (int k=0; k<ima1.zsize(); k++) {
    // Loop over all colums/rows
    for (int ij=0; ij<((pex) ? ima1.ysize() : ima1.xsize()); ij++) {
      bool row_col_is_ok = true;
      if (pex) {
        if (!dv1.RowIsAlright(mask,k,ij)) row_col_is_ok = false;
	else {
	  dv1.SetFromRow(field1[0],k,ij);
	  dv2.SetFromRow(field2[0],k,ij);
	}
      }
      else {
        if (!dv1.ColumnIsAlright(mask,k,ij)) row_col_is_ok = false;
	else {
	  dv1.SetFromColumn(field1[1],k,ij);
	  dv2.SetFromColumn(field2[1],k,ij);
	}
      }
      if (row_col_is_ok) {
	NEWMAT::Matrix K = dv1.GetK_Matrix(sf1) & dv2.GetK_Matrix(sf2);
	NEWMAT::Matrix KtK = K.t()*K + StS;
	NEWMAT::ColumnVector y;
        if (pex) y = ima1.ExtractRow(ij,k) & ima2.ExtractRow(ij,k);
	else y = ima1.ExtractColumn(ij,k) & ima2.ExtractColumn(ij,k);
	NEWMAT::ColumnVector Kty = K.t()*y;
	NEWMAT::ColumnVector y_hat = KtK.i()*Kty;
	if (pex) _rvol.SetRow(ij,k,y_hat); else _rvol.SetColumn(ij,k,y_hat);
      }
      else {
	if (pex) _omask.SetRow(ij,k,zeros); else _omask.SetColumn(ij,k,zeros);
      }
    }
  }
  return;
} EddyCatch

} // End namespace EDDY
