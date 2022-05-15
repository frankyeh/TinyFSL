/*! \file b0Predictor.h
    \brief Contains declaration of class for making silly (mean) predictions about b=0 data.

    \author Jesper Andersson
    \version 1.0b, Sep., 2012.
*/
// Declarations of class to make silly
// predictions about b0 scans.
//
// b0Predictor.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford
//

#ifndef b0Predictor_h
#define b0Predictor_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"
#include "DWIPredictionMaker.h"

namespace EDDY {

/****************************************************************//**
*
* \brief Class for making silly (mean) predictions about b=0 data.
*
* Will make predictions about b=0 scans. Since we don't expect any
* variability in these scans the prediction will simply be the mean
* of all the scans in the prediction maker. It is there mainly so
* that the same code can be used for registration of
* diffusion-weighted and b=0 scans.
*
********************************************************************/
  class b0Predictor : public DWIPredictionMaker
{
public:
  b0Predictor() EddyTry : _pop(true), _valid(false) {} EddyCatch
  ~b0Predictor() {}
  virtual NEWIMAGE::volume<float> Predict(unsigned int indx, bool exclude=false) const EddyTry { return(predict()); } EddyCatch
  virtual NEWIMAGE::volume<float> Predict(unsigned int indx, bool exclude=false) EddyTry { return(predict()); } EddyCatch
  virtual NEWIMAGE::volume<float> PredictCPU(unsigned int indx, bool exclude=false) EddyTry { return(predict()); } EddyCatch
  virtual std::vector<NEWIMAGE::volume<float> > Predict(const std::vector<unsigned int>& indicies, bool exclude=false);
  /// Returns input data for point given by indx
  virtual NEWIMAGE::volume<float> InputData(unsigned int indx) const;
  /// Returns input data for points given by indicies
  virtual std::vector<NEWIMAGE::volume<float> > InputData(const std::vector<unsigned int>& indicies) const;
  /// Returns variance of prediction for point given by indx.
  virtual double PredictionVariance(unsigned int indx, bool exclude=false) EddyTry { return(_mean.mean(_mask) / std::sqrt(double(_sptrs.size()))); } EddyCatch
  /// Returns measurement-error variance for point given by indx.
  virtual double ErrorVariance(unsigned int indx) const EddyTry { return(_mean.mean(_mask)); } EddyCatch
  virtual bool IsPopulated() const;                  // Returns true if all data present
  virtual bool IsValid() const { return(_valid); }   // Returns true if ready to make predictions
  virtual void SetNoOfScans(unsigned int n);
  virtual void SetScan(const NEWIMAGE::volume<float>& scan, // _May_ be thread safe if used "sensibly"
		       const DiffPara&                dp,
		       unsigned int                   indx);
  virtual unsigned int NoOfHyperPar() const { return(0); }
  virtual std::vector<double> GetHyperPar() const EddyTry { std::vector<double> rval; return(rval); } EddyCatch
  virtual void EvaluateModel(const NEWIMAGE::volume<float>& mask, float fwhm, bool verbose=false) EddyTry
  {
    if (!IsPopulated()) throw EddyException("b0Predictor::EvaluateModel: Not ready to evaluate model");
    if (!IsValid()) { _mask=mask; _mean=mean_vol(_mask); _var=variance_vol(_mean,_mask); }
  } EddyCatch
  virtual void EvaluateModel(const NEWIMAGE::volume<float>& mask, bool verbose=false) EddyTry { EvaluateModel(mask,0.0,verbose); } EddyCatch
  virtual void WriteImageData(const std::string& fname) const;
  virtual void WriteMetaData(const std::string& fname) const {}
  virtual void Write(const std::string& fname) const EddyTry { WriteImageData(fname); WriteMetaData(fname); } EddyCatch

private:
  std::vector<std::shared_ptr<NEWIMAGE::volume<float> > > _sptrs; // Pointers to the scans
  NEWIMAGE::volume<float>                                 _mean;  // Mean image
  NEWIMAGE::volume<float>                                 _var;   // Variance image
  NEWIMAGE::volume<float>                                 _mask;  // Mask indicating valid voxels
  mutable bool                                            _pop;   // Tells if all data is present
  bool                                                    _valid; // Tells if it is ready to make predictions

  const NEWIMAGE::volume<float>& predict() const EddyTry
  {
    if (!IsValid()) throw EddyException("b0Predictor::Predict: Not ready to make predictions");
    return(_mean);
  } EddyCatch

  NEWIMAGE::volume<float> mean_vol(const NEWIMAGE::volume<float>& mask) EddyTry
  {
    NEWIMAGE::volume<float> mvol = *_sptrs[0];
    mvol = 0.0;
    for (int k=0; k<_sptrs[0]->zsize(); k++) {
      for (int j=0; j<_sptrs[0]->ysize(); j++) {
	for (int i=0; i<_sptrs[0]->xsize(); i++) {
	  if (1) {
	    // if (mask(i,j,k)) { I removed masking 13/5-2016. J.A.
            float &v = mvol(i,j,k);
	    for (unsigned int s=0; s<_sptrs.size(); s++) {
              v += (*_sptrs[s])(i,j,k);
	    }
            v /= float(_sptrs.size());
	  }
	}
      }
    }
    _valid = true;
    return(mvol);
  } EddyCatch

  NEWIMAGE::volume<float> variance_vol(const NEWIMAGE::volume<float>& mean,
				       const NEWIMAGE::volume<float>& mask) EddyTry
  {
    NEWIMAGE::volume<float> vvol = *_sptrs[0];
    vvol = 0.0;
    if (_sptrs.size() > 1) {
      for (int k=0; k<_sptrs[0]->zsize(); k++) {
	for (int j=0; j<_sptrs[0]->ysize(); j++) {
	  for (int i=0; i<_sptrs[0]->xsize(); i++) {
	    if (1) {
	      // if (mask(i,j,k)) { I removed masking 13/5-2016. J.A.
	      float &v = vvol(i,j,k);
	      float mv = mean(i,j,k);
	      for (unsigned int s=0; s<_sptrs.size(); s++) {
		v += MISCMATHS::Sqr(((*_sptrs[s])(i,j,k) - mv));
	      }
	      v /= float(_sptrs.size()-1);
	    }
	  }
	}
      }
    }
    _valid = true;
    return(vvol);
  } EddyCatch
};

} // End namespace EDDY

#endif // End #ifndef b0Predictor_h
