/*! \file DiffusionGP.cpp
    \brief Contains definitions for class for making Gaussian process based predictions about DWI data.

    \author Jesper Andersson
    \version 1.0b, Sep., 2013.
*/
// Definitions of class to make Gaussian-Process
// based predictions about diffusion data.
//
// DiffusionGP.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2013 University of Oxford
//

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"
#include "EddyUtils.h"
#include "KMatrix.h"
#include "HyParEstimator.h"
#include "DiffusionGP.h"

using namespace EDDY;

/****************************************************************//**
*
* Constructs a DiffusionGP object given files containing input data.
* When using this constructor the object is immediately ready to do
* predictions.
* \param scans_fname Name of file containing multiple diffusion
*        weighted volumes.
* \param var_mask_fname
* \param dpars Vector of objects of DiffPara type that specifies
*        diffusion weighting and direction for the volumes in
*        scans_fname.
*
*
********************************************************************/
DiffusionGP::DiffusionGP(const std::shared_ptr<const KMatrix>&        Kmat,
			 const std::shared_ptr<const HyParEstimator>& hpe,
			 const std::string&                           scans_fname,
			 const std::string&                           brain_mask_fname,
                         const std::vector<DiffPara>&                 dpars,
			 float                                        fwhm,
			 bool                                         verbose) EddyTry
: _Kmat(Kmat->Clone()), _hpe(hpe->Clone()), _pop(false), _mc(false)
{
  NEWIMAGE::volume4D<float> scans;
  EddyUtils::read_DWI_volume4D(scans,scans_fname,dpars);
  for (int s=0; s<scans.tsize(); s++) _sptrs.push_back(std::shared_ptr<NEWIMAGE::volume<float> >(new NEWIMAGE::volume<float>(scans[s])));
  _pop = true;
  _dpars = EddyUtils::GetDWIDiffParas(dpars);
  NEWIMAGE::volume<float> brain_mask; NEWIMAGE::read_volume(brain_mask,brain_mask_fname);
  _Kmat->SetDiffusionPar(_dpars);
  std::vector<std::vector<unsigned int> > mi = _Kmat->GetMeanIndicies();
  mean_correct(mi);
  DataSelector vd(_sptrs,brain_mask,_hpe->GetNVox(),fwhm,_hpe->RndInit());
  _hpe->SetData(vd.GetData());
  _hpe->Estimate(_Kmat,verbose);
  _Kmat->SetHyperPar(hpe->GetHyperParameters());
  _Kmat->CalculateInvK();
} EddyCatch

NEWIMAGE::volume<float> DiffusionGP::Predict(unsigned int indx,
					     bool         exclude) EddyTry
{
  if (!IsPopulated()) throw EddyException("DiffusionGP::Predict:non-const: Not yet fully populated");
  if (!IsValid()) throw EddyException("DiffusionGP::Predict:non-const: Not yet ready for predictions");
  NEWMAT::RowVector pv = _Kmat->PredVec(indx,exclude); // Calls non-const version
  NEWIMAGE::volume<float> pi = *_sptrs[0]; pi = 0.0;
  #ifdef COMPILE_GPU
  predict_image_gpu(indx,exclude,pv,pi);
  #else
  predict_image_cpu(indx,exclude,pv,pi);
  #endif
  return(pi);
} EddyCatch

NEWIMAGE::volume<float> DiffusionGP::PredictCPU(unsigned int indx,
						bool         exclude) EddyTry
{
  if (!IsPopulated()) throw EddyException("DiffusionGP::Predict:non-const: Not yet fully populated");
  if (!IsValid()) throw EddyException("DiffusionGP::Predict:non-const: Not yet ready for predictions");
  NEWMAT::RowVector pv = _Kmat->PredVec(indx,exclude); // Calls non-const version
  NEWIMAGE::volume<float> pi = *_sptrs[0]; pi = 0.0;
  predict_image_cpu(indx,exclude,pv,pi);
  return(pi);
} EddyCatch

NEWIMAGE::volume<float> DiffusionGP::Predict(unsigned int indx,
					     bool         exclude) const EddyTry
{
  if (!IsPopulated()) throw EddyException("DiffusionGP::Predict:const: Not yet fully populated");
  if (!IsValid()) throw EddyException("DiffusionGP::Predict:const: Not yet ready for predictions");
  NEWMAT::RowVector pv = _Kmat->PredVec(indx,exclude); // Calls const version
  NEWIMAGE::volume<float> pi = *_sptrs[0]; pi = 0.0;
  #ifdef COMPILE_GPU
  predict_image_gpu(indx,exclude,pv,pi);
  #else
  predict_image_cpu(indx,exclude,pv,pi);
  #endif
  return(pi);
} EddyCatch

std::vector<NEWIMAGE::volume<float> > DiffusionGP::Predict(const std::vector<unsigned int>& indicies,
							   bool                             exclude) EddyTry
{
  if (!IsPopulated()) throw EddyException("DiffusionGP::Predict: Not yet fully populated");
  if (!IsValid()) throw EddyException("DiffusionGP::Predict: Not yet ready for predictions");
  std::vector<NEWIMAGE::volume<float> > pi(indicies.size());
  std::vector<NEWMAT::RowVector> pvecs(indicies.size());
  for (unsigned int i=0; i<indicies.size(); i++) pvecs[i] = _Kmat->PredVec(indicies[i],exclude);
  #ifdef COMPILE_GPU
  predict_images_gpu(indicies,exclude,pvecs,pi);
  #else
  predict_images_cpu(indicies,exclude,pvecs,pi);
  #endif
  return(pi);
} EddyCatch

NEWIMAGE::volume<float> DiffusionGP::InputData(unsigned int indx) const EddyTry
{
  if (!IsPopulated()) throw EddyException("DiffusionGP::InputData: Not yet fully populated");
  if (indx >= _sptrs.size()) throw EddyException("DiffusionGP::InputData: indx out of range");
  return(*(_sptrs[indx]) + *(_mptrs[which_mean(indx)]));
} EddyCatch

std::vector<NEWIMAGE::volume<float> > DiffusionGP::InputData(const std::vector<unsigned int>& indx) const EddyTry
{
  if (!IsPopulated()) throw EddyException("DiffusionGP::InputData: Not yet fully populated");
  for (unsigned int i=0; i<indx.size(); i++) if (indx[i] >= _sptrs.size()) throw EddyException("DiffusionGP::InputData: indx out of range");
  std::vector<NEWIMAGE::volume<float> > rval(indx.size());
  for (unsigned int i=0; i<indx.size(); i++) rval[i] = *(_sptrs[indx[i]]) + *(_mptrs[which_mean(indx[i])]);
  return(rval);
} EddyCatch

double DiffusionGP::PredictionVariance(unsigned int indx,
				       bool         exclude) EddyTry
{
  if (!IsPopulated()) throw EddyException("DiffusionGP::PredictionVariance:const: Not yet fully populated");
  if (!IsValid()) throw EddyException("DiffusionGP::PredictionVariance:const: Not yet ready for predictions");
  double pv = _Kmat->PredVar(indx,exclude);
  return(pv);
} EddyCatch

double DiffusionGP::ErrorVariance(unsigned int indx) const EddyTry
{
  if (!IsPopulated()) throw EddyException("DiffusionGP::ErrorVariance:const: Not yet fully populated");
  if (!IsValid()) throw EddyException("DiffusionGP::ErrorVariance:const: Not yet ready for predictions");
  double ev = _Kmat->ErrVar(indx);
  return(ev);
} EddyCatch

void DiffusionGP::SetNoOfScans(unsigned int n) EddyTry
{
  if (n == _sptrs.size()) return; // No change
  else if (n > _sptrs.size()) {   // If increasing size
    _sptrs.resize(n,std::shared_ptr<NEWIMAGE::volume<float> >()); // New elements populated by NULL
    _dpars.resize(n); // New elements populated with (arbitrary) [1 0 0] bvec
    _Kmat->Reset();
    _pop = false;
  }
  else { // Decreasing size not allowed
    throw EddyException("DiffusionGP::SetNoOfScans: Decreasing size not allowed");
  }
  return;
} EddyCatch

void DiffusionGP::SetScan(const NEWIMAGE::volume<float>& scan,
                          const DiffPara&                dp,
                          unsigned int                   indx) EddyTry
{
  if (int(indx) > (int(_sptrs.size())-1)) throw EddyException("DiffusionGP::SetScan: Invalid image index");
  if (_sptrs.size() && _sptrs[0] && !NEWIMAGE::samesize(*_sptrs[0],scan)) throw EddyException("DiffusionGP::SetScan: Wrong image dimension");
  if (_sptrs[indx] && dp != _dpars[indx]) throw EddyException("DiffusionGP::SetScan: You cannot change shell or direction of scan");
  _sptrs[indx] = std::shared_ptr<NEWIMAGE::volume<float> >(new NEWIMAGE::volume<float>(scan));
  _dpars[indx] = dp;
  _pop = is_populated();
} EddyCatch

void DiffusionGP::EvaluateModel(const NEWIMAGE::volume<float>& mask,
				float                          fwhm,
				bool                           verbose) EddyTry
{
  if (!IsPopulated()) throw EddyException("DiffusionGP::EvaluateModel: Predictor not fully populated");
  _Kmat->Reset(); _Kmat->SetDiffusionPar(_dpars);
  std::vector<std::vector<unsigned int> > mi = _Kmat->GetMeanIndicies();
  mean_correct(mi);
  DataSelector vd(_sptrs,mask,_hpe->GetNVox(),fwhm,_hpe->RndInit());
  _hpe->SetData(vd.GetData());
  _hpe->Estimate(_Kmat,verbose);
  _Kmat->SetHyperPar(_hpe->GetHyperParameters());
  _Kmat->CalculateInvK();
} EddyCatch

void DiffusionGP::WriteImageData(const std::string& fname) const EddyTry
{
  char ofname[256];
  if (!IsPopulated()) throw EddyException("DiffusionGP::WriteImageData: Not yet fully populated");
  // For practical reasons the volumes are written individually
  for (unsigned int i=0; i<_sptrs.size(); i++) {
    sprintf(ofname,"%s_%03d",fname.c_str(),i);
    NEWIMAGE::write_volume(*(_sptrs[i]),ofname);
  }
  for (unsigned int i=0; i<_mptrs.size(); i++) {
    sprintf(ofname,"%s_mean_%03d",fname.c_str(),i);
    NEWIMAGE::write_volume(*(_mptrs[i]),ofname);
  }
} EddyCatch

void DiffusionGP::mean_correct(const std::vector<std::vector<unsigned int> >& mi) EddyTry
{
  if (!_mc) {
    // First, calculate mean images
    if (_mptrs.size() != mi.size()) _mptrs.resize(mi.size());
    for (unsigned int m=0; m<mi.size(); m++) {
      if (_mptrs[m]==nullptr || !samesize(*_sptrs[0],*_mptrs[m])) {
	_mptrs[m] = std::shared_ptr<NEWIMAGE::volume<float> >(new NEWIMAGE::volume<float>(*_sptrs[0]));
      }
      *_mptrs[m] = 0.0;
      for (unsigned int s=0; s<mi[m].size(); s++) {
	*_mptrs[m] += *_sptrs[mi[m][s]];
      }
      *_mptrs[m] /= mi[m].size();
    }
    // Next, correct individual images
    for (unsigned int m=0; m<mi.size(); m++) {
      for (unsigned int s=0; s<mi[m].size(); s++) {
	*_sptrs[mi[m][s]] -= *_mptrs[m];
      }
    }
    _mc = true;
  }
} EddyCatch

bool DiffusionGP::is_populated() const EddyTry
{
  bool pop = true;
  for (unsigned int i=0; i<_sptrs.size(); i++) {
    if (!_sptrs[i]) { pop = false; break; }
  }
  return(pop);
} EddyCatch

void DiffusionGP::predict_image_cpu(// Input
				    unsigned int             indx,
				    bool                     exclude,
				    const NEWMAT::RowVector& pv,
				    // Output
				    NEWIMAGE::volume<float>& pi) const EddyTry
{
  unsigned int mi = which_mean(indx);
  unsigned int ys = (exclude) ? _sptrs.size()-1 : _sptrs.size();
  NEWMAT::ColumnVector y(ys);
  pi = *_mptrs[mi];
  for (int k=0; k<pi.zsize(); k++) {
    for (int j=0; j<pi.ysize(); j++) {
      for (int i=0; i<pi.xsize(); i++) {
	if (get_y(i,j,k,indx,exclude,y)) pi(i,j,k) += static_cast<float>((pv*y).AsScalar());
	else pi(i,j,k) = 0.0;
      }
    }
  }
} EddyCatch

void DiffusionGP::predict_images_cpu(// Input
				     const std::vector<unsigned int>&       indicies,
				     bool                                   exclude,
				     const std::vector<NEWMAT::RowVector>&  pvecs,
				     // Output
				     std::vector<NEWIMAGE::volume<float> >& pi) const EddyTry
{
  if (indicies.size() != pvecs.size() || indicies.size() != pi.size()) {
    throw EDDY::EddyException("DiffusionGP::predict_images_cpu: mismatch among indicies, pvecs and pi");
  }
  for (unsigned int i=0; i<indicies.size(); i++) predict_image_cpu(indicies[i],exclude,pvecs[i],pi[i]);
  return;
} EddyCatch

unsigned int DiffusionGP::which_mean(unsigned int indx) const EddyTry
{
  std::vector<std::vector<unsigned int> > mis = _Kmat->GetMeanIndicies();
  if (mis.size() == 1) return(0);
  else {
    for (unsigned int m=0; m<mis.size(); m++) {
      unsigned int j;
      for (j=0; j<mis[m].size(); j++) if (mis[m][j] == indx) break;
      if (j<mis[m].size()) return(m);
    }
    throw EddyException("DiffusionGP::which_mean: Invalid indx");
  }
} EddyCatch

bool DiffusionGP::get_y(// Input
			unsigned int           i,
			unsigned int           j,
			unsigned int           k,
			unsigned int           indx,
			bool                   exclude,
			// Output
			NEWMAT::ColumnVector&  y) const EddyTry
{
  for (unsigned int t=0, tt=1; t<_sptrs.size(); t++) {
    if (!exclude || t!=indx) {
      if (!(*_sptrs[t])(i,j,k)) return(false);
      else y(tt++) = (*_sptrs[t])(i,j,k);
    }
  }
  return(true);
} EddyCatch
