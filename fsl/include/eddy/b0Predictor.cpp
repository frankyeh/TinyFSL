// Definitions of class to make silly
// predictions about b0 scans.
//
// b0Predictor.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford
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
#include "b0Predictor.h"

using namespace EDDY;

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class b0Predictor
//
//
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

std::vector<NEWIMAGE::volume<float> > b0Predictor::Predict(const std::vector<unsigned int>& indicies,
							   bool                             exclude) EddyTry
{
  std::vector<NEWIMAGE::volume<float> > rval(indicies.size());
  for (unsigned int i=0; i<indicies.size(); i++) rval[i]  = predict();
  return(rval);
} EddyCatch

NEWIMAGE::volume<float> b0Predictor::InputData(unsigned int indx) const EddyTry
{
  if (!IsPopulated()) throw EddyException("b0Predictor::InputData: Not yet fully populated");
  if (indx >= _sptrs.size()) throw EddyException("b0Predictor::InputData: indx out of range");
  return(*(_sptrs[indx]));
} EddyCatch

std::vector<NEWIMAGE::volume<float> > b0Predictor::InputData(const std::vector<unsigned int>& indx) const EddyTry
{
  if (!IsPopulated()) throw EddyException("b0Predictor::InputData: Not yet fully populated");
  for (unsigned int i=0; i<indx.size(); i++) if (indx[i] >= _sptrs.size()) throw EddyException("b0Predictor::InputData: indx out of range");
  std::vector<NEWIMAGE::volume<float> > rval(indx.size());
  for (unsigned int i=0; i<indx.size(); i++) rval[i] = *(_sptrs[indx[i]]);
  return(rval);
} EddyCatch

bool b0Predictor::IsPopulated() const EddyTry
{
  if (_pop) return(true);
  else {
    _pop = true;
    for (unsigned int i=0; i<_sptrs.size(); i++) {
      if (!_sptrs[i]) { _pop = false; break; }
    }
  }
  return(_pop);
} EddyCatch

void b0Predictor::SetNoOfScans(unsigned int n) EddyTry
{
  if (n == _sptrs.size()) return; // No change
  else if (n > _sptrs.size()) {   // If increasing size
    _sptrs.resize(n,std::shared_ptr<NEWIMAGE::volume<float> >()); // New elements populated by NULL
    _pop = false;
    _valid = false;
  }
  else { // If decreasing size
    _sptrs.resize(n);
    _valid = false;
    if (_pop==false) { // _pop may potentially go from false to true
      _pop = IsPopulated();
    }
  }
  return;
} EddyCatch

void b0Predictor::SetScan(const NEWIMAGE::volume<float>& scan,
                          const DiffPara&                dp,
                          unsigned int                   indx) EddyTry
{
  if (int(indx) > (int(_sptrs.size())-1)) throw EddyException("b0Predictor::SetScan: Invalid image index");
  if (_sptrs.size() && _sptrs[0] && !NEWIMAGE::samesize(*_sptrs[0],scan)) throw EddyException("b0Predictor::SetScan: Wrong image dimension");
  _sptrs[indx] = std::shared_ptr<NEWIMAGE::volume<float> >(new NEWIMAGE::volume<float>(scan));
} EddyCatch

void b0Predictor::WriteImageData(const std::string& fname) const EddyTry
{
  char ofname[256];
  if (!IsPopulated()) throw EddyException("DiffusionGP::Write: Not yet fully populated");
  // For practical reasons the volumes are written individually
  for (unsigned int i=0; i<_sptrs.size(); i++) {
    sprintf(ofname,"%s_%03d",fname.c_str(),i);
    NEWIMAGE::write_volume(*(_sptrs[i]),ofname);
  }
  NEWIMAGE::write_volume(_mean,fname+std::string("_mean"));
  NEWIMAGE::write_volume(_var,fname+std::string("_var"));
  NEWIMAGE::write_volume(_mask,fname+std::string("_mask"));
} EddyCatch
