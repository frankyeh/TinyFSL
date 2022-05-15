/*! \file DiffusionGP.cu
    \brief Contains definitions for class for making Gaussian process based predictions about DWI data.

    \author Jesper Andersson
    \version 1.0b, Feb., 2013.
*/
// Definitions of class to make Gaussian-Process
// based predictions about diffusion data.
//
// DiffusionGP.cu
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford
//

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#pragma push
#pragma diag_suppress = code_is_unreachable // Supress warnings from armawrap
#pragma diag_suppress = expr_has_no_effect  // Supress warnings from boost
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#pragma pop
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"
#include "EddyUtils.h"
#include "DiffusionGP.h"
#include "CudaVolume.h"

using namespace EDDY;

/****************************************************************//**
* \brief Returns prediction for point given by indx
*
* Returns a predicted image for the given index, where index refers
* to the corresponding bvec (i.e. bvecs[index]).
* except that the noise variance is given as a parameter (rather than
* using that which is stored in the object).
* It should be noted that this implementation isn't very efficient as
* it reloads all image to the GPU. That means that if there are N images
* and we want to predict them all it takes N*N transfers to the GPU.
* \param index refers to the corresponding bvec (i.e. bvecs[index]).
* \param exclude Decides if indx itself should be used in the prediction
* (exclude=false) or not (exclude=true)
* \param pvec Prediction vector for index
* \param pi The "predicted image"
*
********************************************************************/
void DiffusionGP::predict_image_gpu(// Input
				    unsigned int             indx,
				    bool                     exclude,
				    const NEWMAT::RowVector& pvec,
				    // Output
				    NEWIMAGE::volume<float>& pi) const EddyTry
{
  if (!NEWIMAGE::samesize(pi,*_sptrs[0])) {
    pi.reinitialize(_sptrs[0]->xsize(),_sptrs[0]->ysize(),_sptrs[0]->zsize());
    NEWIMAGE::copybasicproperties(*_sptrs[0],pi);
  }
  EDDY::CudaVolume pcv(pi,false);
  for (unsigned int s=0; s<_sptrs.size(); s++) {
    // Next row shows what the function below does (a little faster)
    // pcv += pvec(i+1) * EDDY::CudaVolume(*(_sptrs[s]));
    if (exclude) {
      if (s < indx) pcv.MultiplyAndAddToMe(EDDY::CudaVolume(*(_sptrs[s])),pvec(s+1));
      // Do nothing if (s == indicies[i])
      else if (s > indx) pcv.MultiplyAndAddToMe(EDDY::CudaVolume(*(_sptrs[s])),pvec(s));
    }
    else pcv.MultiplyAndAddToMe(EDDY::CudaVolume(*(_sptrs[s])),pvec(s+1));
  }
  pcv += EDDY::CudaVolume(*_mptrs[which_mean(indx)]);
  pcv.GetVolume(pi);
  return;
} EddyCatch

void DiffusionGP::predict_images_gpu(// Input
				     const std::vector<unsigned int>&       indicies,
				     bool                                   exclude,
				     const std::vector<NEWMAT::RowVector>&  pvecs,
				     // Output
				     std::vector<NEWIMAGE::volume<float> >& pi) const EddyTry
{
  if (indicies.size() != pvecs.size() || indicies.size() != pi.size()) {
    throw EDDY::EddyException("DiffusionGP::predict_images_gpu: mismatch among indicies, pvecs and pi");
  }
  // Start by allocating space on GPU for all output images
  std::vector<EDDY::CudaVolume> pcvs(indicies.size());
  for (unsigned int i=0; i<indicies.size(); i++) {
    if (!NEWIMAGE::samesize(pi[i],*_sptrs[0])) {
      pi[i].reinitialize(_sptrs[0]->xsize(),_sptrs[0]->ysize(),_sptrs[0]->zsize());
      NEWIMAGE::copybasicproperties(*_sptrs[0],pi[i]);
    }
    pcvs[i].SetHdr(pi[i]);
  }
  // Transfer all mean images to the GPU
  std::vector<EDDY::CudaVolume> means(_mptrs.size());
  for (unsigned int m=0; m<means.size(); m++) means[m] = *(_mptrs[m]);
  // Do the GP predictions
  for (unsigned int s=0; s<_sptrs.size(); s++) { // s index into original volumes
    EDDY::CudaVolume cv = *(_sptrs[s]);
    for (unsigned int i=0; i<indicies.size(); i++) { // i index into predictions
      if (exclude) {
	if (s < indicies[i]) pcvs[i].MultiplyAndAddToMe(cv,pvecs[i](s+1));
	// Do nothing if (s == indicies[i])
	else if (s > indicies[i]) pcvs[i].MultiplyAndAddToMe(cv,pvecs[i](s));
      }
      else pcvs[i].MultiplyAndAddToMe(cv,pvecs[i](s+1));
    }
  }
  // Add means to predictions and transfer back from GPU
  for (unsigned int i=0; i<indicies.size(); i++) {
    pcvs[i] += means[which_mean(indicies[i])];
    pcvs[i].GetVolume(pi[i]);
  }
  return;
} EddyCatch
