/*! \file DiffusionGP.h
    \brief Contains declaration of class for making Gaussian process based predictions about DWI data.

    \author Jesper Andersson
    \version 1.0b, Sep., 2012.
*/
// Declarations of class to make Gaussian-Process
// based predictions about diffusion data.
//
// DiffusionGP.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford
//

#ifndef DiffusionGP_h
#define DiffusionGP_h

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
#include "KMatrix.h"
#include "HyParEstimator.h"

namespace EDDY {

/****************************************************************//**
*
* \brief Class used to make Gaussian process based predictions
* about diffusion data.
*
* Will make predictions about observed (smoothing) and unobserved
* (interpolation) scans using Gaussian Processes. It works by first
* creating an object, setting the # of scans one wants it to contain
* and then populate it with scans. Once that is done it is ready
* to start making predictions. If you think of the signal (in a
* given voxel) as points on a surface the Gaussian process will make
* predictions from the assumption that that surface should be smooth.
*
********************************************************************/
class DiffusionGP : public DWIPredictionMaker
{
public:
  /// Default constructor
  DiffusionGP(const std::shared_ptr<const KMatrix>&        Kmat,
	      const std::shared_ptr<const HyParEstimator>& hpe) EddyTry : _Kmat(Kmat->Clone()), _hpe(hpe->Clone()), _pop(true), _mc(false) {} EddyCatch
  /// Constructor that takes filenames from which to load data
  DiffusionGP(const std::shared_ptr<const KMatrix>&        Kmat,
	      const std::shared_ptr<const HyParEstimator>& hpe,
	      const std::string&                           scans_fname,
              const std::string&                           var_mask_fname,
              const std::vector<DiffPara>&                 dpars,
	      float                                        fwhm=0.0,
	      bool                                         verbose=false);
  /// Returns prediction for point given by indx.
  virtual NEWIMAGE::volume<float> Predict(unsigned int indx, bool exclude=false) const;
  /// Returns prediction for point given by indx.
  virtual NEWIMAGE::volume<float> Predict(unsigned int indx, bool exclude=false);
  /// Returns prediction for point given by indx. This is used only as a means of directly comparing CPU and GPU outputs.
  virtual NEWIMAGE::volume<float> PredictCPU(unsigned int indx, bool exclude=false);
  /// Returns predictions for points given by indicies
  virtual std::vector<NEWIMAGE::volume<float> > Predict(const std::vector<unsigned int>& indicies, bool exclude=false);
  /// Returns input data for point given by indx
  virtual NEWIMAGE::volume<float> InputData(unsigned int indx) const;
  /// Returns input data for points given by indicies
  virtual std::vector<NEWIMAGE::volume<float> > InputData(const std::vector<unsigned int>& indicies) const;
  /// Returns variance of prediction for point given by indx.
  virtual double PredictionVariance(unsigned int indx, bool exclude=false);
  /// Returns measurement-error variance for point given by indx.
  virtual double ErrorVariance(unsigned int indx) const;
  /// Returns true if all data has been loaded
  virtual bool IsPopulated() const { return(_pop); }
  /// Indicates if it is ready to make predictions.
  virtual bool IsValid() const EddyTry { return(IsPopulated() && _mc && _Kmat->IsValid()); } EddyCatch
  /// Specify the # of points we plan to put into the predictor.
  virtual void SetNoOfScans(unsigned int n);
  /// Set a point given by indx. This function is thread safe as long as different threads set different points.
  virtual void SetScan(const NEWIMAGE::volume<float>& scan,  // May be thread safe if used "sensibly"
		       const DiffPara&                dp,
		       unsigned int                   indx);
  /// Returns the number of hyperparameters for the model
  virtual unsigned int NoOfHyperPar() const EddyTry { return(_Kmat->NoOfHyperPar()); } EddyCatch
  /// Returns the hyperparameters for the model
  virtual std::vector<double> GetHyperPar() const EddyTry { return(_Kmat->GetHyperPar()); } EddyCatch
  /// Evaluates the model so as to make the predictor ready to make predictions.
  virtual void EvaluateModel(const NEWIMAGE::volume<float>& mask, float fwhm, bool verbose=false);
  /// Evaluates the model so as to make the predictor ready to make predictions.
  virtual void EvaluateModel(const NEWIMAGE::volume<float>& mask, bool verbose=false) EddyTry { EvaluateModel(mask,0.0,verbose); } EddyCatch
  /// Writes internal content to disk for debug purposes
  virtual void WriteImageData(const std::string& fname) const;
  virtual void WriteMetaData(const std::string& fname) const EddyTry {
    if (!IsPopulated()) throw EddyException("DiffusionGP::WriteMetaData: Not yet fully populated"); _Kmat->Write(fname);
  } EddyCatch
  virtual void Write(const std::string& fname) const EddyTry { WriteImageData(fname); WriteMetaData(fname); } EddyCatch
private:
  static const unsigned int nvoxhp = 500;

  std::vector<std::shared_ptr<NEWIMAGE::volume<float> > > _sptrs;   // Pointers to the scans
  std::shared_ptr<KMatrix>                                _Kmat;    // K-matrix for Gaussian Process
  std::vector<DiffPara>                                   _dpars;   // Diffusion parameters
  std::shared_ptr<HyParEstimator>                         _hpe;     // Hyperparameter estimator
  bool                                                    _pop;     // True if all data is present
  bool                                                    _mc;      // True if data has been mean-corrected
  std::vector<std::shared_ptr<NEWIMAGE::volume<float> > > _mptrs;   // Pointers to mean images

  void mean_correct(const std::vector<std::vector<unsigned int> >& mi);
  bool is_populated() const;
  void predict_image_cpu(unsigned int             indx,
			 bool                     excl,
			 const NEWMAT::RowVector& pv,
			 NEWIMAGE::volume<float>& ima) const;
  void predict_images_cpu(// Input
			  const std::vector<unsigned int>&       indicies,
			  bool                                   exclude,
			  const std::vector<NEWMAT::RowVector>&  pvecs,
			  // Output
			  std::vector<NEWIMAGE::volume<float> >& pi) const;
  #ifdef COMPILE_GPU
  void predict_image_gpu(unsigned int             indx,
			 bool                     excl,
			 const NEWMAT::RowVector& pv,
			 NEWIMAGE::volume<float>& ima) const;
  void predict_images_gpu(// Input
			  const std::vector<unsigned int>&       indicies,
			  bool                                   exclude,
			  const std::vector<NEWMAT::RowVector>&  pvecs,
			  // Output
			  std::vector<NEWIMAGE::volume<float> >& pi) const;
  #endif
  unsigned int which_mean(unsigned int indx) const;

  bool get_y(// Input
	     unsigned int           i,
	     unsigned int           j,
	     unsigned int           k,
	     unsigned int           indx,
	     bool                   exclude,
	     // Output
	     NEWMAT::ColumnVector&  y) const;
};

} // End namespace EDDY

#endif // End #ifndef DiffusionGP_h
