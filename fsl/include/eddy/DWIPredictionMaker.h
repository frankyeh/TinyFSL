/*! \file DWIPredictionMaker.h
    \brief Contains declaration of virtual base class for making predictions about DWI data.

    \author Jesper Andersson
    \version 1.0b, Sep., 2012.
*/
// Declarations of virtual base class for
// making predictions about DWI data.
//
// DWIPredictionMaker.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford
//

#ifndef DWIPredictionMaker_h
#define DWIPredictionMaker_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"

namespace EDDY {


/****************************************************************//**
*
* \brief Virtual base class for classes used to make predictions
* about diffusion data.
*
* The idea of the prediction makers is to be able to provide them
* with some set of data (for some set of diffusion gradients) and
* then use it to make predictions about what a certain data point
* should be. The predictions could be about unobserved points (in
* which case it would perform an interpolation) or about observed
* points (which would amount to smoothing).
* The virtual class provides a minimal interface for actual prediction
* makers. These could be based e.g. on the diffusion tensor model
* or on Gaussian processes.
********************************************************************/
class DWIPredictionMaker
{
public:
  DWIPredictionMaker() {}
  virtual ~DWIPredictionMaker() {}
  /// Returns prediction for point given by indx.
  virtual NEWIMAGE::volume<float> Predict(unsigned int indx, bool exclude=false) const = 0;
  /// Returns prediction for point given by indx.
  virtual NEWIMAGE::volume<float> Predict(unsigned int indx, bool exclude=false) = 0;
  /// Returns prediction for point given by indx. This is used only as a means of directly comparing CPU and GPU outputs.
  virtual NEWIMAGE::volume<float> PredictCPU(unsigned int indx, bool exclude=false) = 0;
  /// Returns predictions for a set of points given by indicies
  virtual std::vector<NEWIMAGE::volume<float> > Predict(const std::vector<unsigned int>& indicies, bool exclude=false) = 0;
  /// Returns input data for point given by indx
  virtual NEWIMAGE::volume<float> InputData(unsigned int indx) const = 0;
  /// Returns input data for points given by indicies
  virtual std::vector<NEWIMAGE::volume<float> > InputData(const std::vector<unsigned int>& indicies) const = 0;
  /// Returns variance of prediction for point given by indx.
  virtual double PredictionVariance(unsigned int indx, bool exclude=false) = 0;
  /// Returns measurement-error variance for point given by indx.
  virtual double ErrorVariance(unsigned int indx) const = 0;
  /// Returns true if all data has been loaded
  virtual bool IsPopulated() const = 0;
  /// Indicates if it is ready to make predictions.
  virtual bool IsValid() const = 0;
  /// Specify the # of points we plan to put into the predictor.
  virtual void SetNoOfScans(unsigned int n) = 0;
  /*
  /// Adds a new point to the end of the current list. This function is NOT thread safe.
  virtual void AddScan(const NEWIMAGE::volume<float>& scan,  // NOT thread safe
                       const DiffPara&                dp) = 0;
  */
  /// Set a point given by indx. This function is thread safe as long as different threads set different points.
  virtual void SetScan(const NEWIMAGE::volume<float>& scan,  // May be thread safe if used "sensibly"
                       const DiffPara&                dp,
                       unsigned int                   indx) = 0;
  /// Returns the number of hyperparameters for the model
  virtual unsigned int NoOfHyperPar() const = 0;
  /// Returns the hyperparameters for the model
  virtual std::vector<double> GetHyperPar() const = 0;
  /// Evaluates the model so as to make the predictor ready to make predictions.
  virtual void EvaluateModel(const NEWIMAGE::volume<float>& mask, bool verbose=false) = 0;
  /// Evaluates the model so as to make the predictor ready to make predictions.
  virtual void EvaluateModel(const NEWIMAGE::volume<float>& mask, float fwhm, bool verbose=false) = 0;
  /// Writes internal content to disk for debug purposes
  virtual void WriteImageData(const std::string& fname) const = 0;
  virtual void WriteMetaData(const std::string& fname) const = 0;
  virtual void Write(const std::string& fname) const = 0;
};

} // End namespace EDDY

#endif // End #ifndef DWIPredictionMaker_h

/*!
  \fn NEWIMAGE::volume<float> DWIPredictionMaker::Predict(unsigned int indx) const = 0
  Returns a prediction for the scan given by indx, where indx pertains to the indx that was used when setting a given
  scan with a call to SetScan. So if for example there was a call PM.SetScan(scan,my_dp,5) then PM.Predict(5) will return
  a prediction for the diffusion weighting specified in my_dp.
  \param indx Specifies which scan, and indirectly what diffusion weighting, we want the prediction for. It is zero-offset so should be in the range 0--n-1 where n has been set by an earlier call to DWIPredictionMaker::SetNoOfScans(unsigned int n).
  \return A predicted image volume.
*/

/*!
  \fn NEWIMAGE::volume<float> DWIPredictionMaker::Predict(const DiffPara& dpar) const = 0
  Returns a prediction for a scan with diffusion parameters given by dpar. This may pertain to an observed scan (smoothing) or an unobserved scan (interpolation).
  \param dpar Specifies the diffusion weighting.
  \return A predicted image volume.
*/

/*!
  \fn NEWIMAGE::volume4D<float> DWIPredictionMaker::PredictAll()
  Returns predictions for all images/diffusion weightings that have been set in the prediction maker.
  \return A 4D volume with as many volumes as there are images in the prediction maker (as set by an earlier call to DWIPredictionMaker::SetNoOfScans(unsigned int n)).
*/

/*!
  \fn bool DWIPredictionMaker::IsPopulated() const
  Returns true if valid scans have been set for all "slots" as
  defined by a call to DiffusionGP::SetNoOfScans(unsigned int).
*/

/*!
  \fn bool DWIPredictionMaker::IsValid() const
  Will return true if the object is ready to make predictions. For this to be true it must have been fully populated.
*/

/*!
  \fn void DWIPredictionMaker::SetNoOfScans(unsigned int n)
  Specifies the number of scans to use for the prediction maker. It can override an earlier call to increase or shrink the number of scans.
  \param n The number of scans
*/

/*!
  \fn void DWIPredictionMaker::AddScan(const NEWIMAGE::volume<float>& scan, const DiffPara& dp)
  Adds a scan to the end of the current list of scans in the prediction maker.
  \param scan The scan that should be added.
  \param dp Diffusion parameters pertaining to scan.
*/

/*!
  \fn void DWIPredictionMaker::SetScan(const NEWIMAGE::volume<float>& scan, const DiffPara& dp, unsigned int indx)
  Inserts a scan in the slot given by indx into the list of scans in the prediction maker.
  \param scan The scan that should be inserted.
  \param dp Diffusion parameters pertaining to scan.
  \param indx Slot in scan list into which to put scan. It is zero-offset so should be in the range 0--n-1 where n has been set by an earlier call to DWIPredictionMaker::SetNoOfScans(unsigned int n).
*/

/*!
  \fn void DWIPredictionMaker::EvaluateModel(const NEWIMAGE::volume<float>& mask)
  Performs the calculations that need to be done after all the data has been loaded and before predictions can be made. The details of this will depend on the specific derived class. If for example the derived class is based on the diffusion tensor model it will entail calculating the diffusion tensor for each voxel.
  \param mask Binary mask used to limit the calculations to where the voxels are set to one in the mask.
*/
