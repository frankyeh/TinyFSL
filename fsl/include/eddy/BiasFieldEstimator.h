/*! \file BiasFieldEstimator.h
    \brief Contains declaration of class for estimation of a bias field

    \author Jesper Andersson
    \version 1.0b, April, 2017.
*/
// 
// BiasFieldEstimator.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2017 University of Oxford 
//

#ifndef BiasFieldEstimator_h
#define BiasFieldEstimator_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <time.h>
#include "newimage/newimageall.h"
#include "ECScanClasses.h"

namespace EDDY {

class BiasFieldEstimatorImpl;

/****************************************************************//**
*
* \brief Class used to estimate a receive bias-field.
*
* Class used to estimate a receive bias-field based on data
* from a subject that moves around within that bias-field. 
* It is implemented using the "Pimpl idiom" which means that this class 
* only implements an interface whereas the actual work is being performed
* by the BiasFieldEstimatorImpl class which is declared and defined in
* BiasFieldEstimatorImpl.cpp or cuda/BiasFieldEstimatorImpl.cu depending on 
* what platform the code is compiled for.
*
* The way you would use it is by first constructing the object and
* then calling `AddScan()` for all volumes and finally call `GetField()`
* to retrieve the estimated field
*
********************************************************************/ 
class BiasFieldEstimator
{
public:
  BiasFieldEstimator();
  ~BiasFieldEstimator();
  /// Set ref scan
  void SetRefScan(const NEWIMAGE::volume<float>& mask, const EDDY::ImageCoordinates& coords);
  /// Add another scan
  void AddScan(const NEWIMAGE::volume<float>& predicted, const NEWIMAGE::volume<float>& observed, 
	       const NEWIMAGE::volume<float>& mask, const EDDY::ImageCoordinates& coords);
  /// Calculate and return "direct" representation of the bias-field
  NEWIMAGE::volume<float> GetField(double lambda) const;
  /// Calculate and return spline basis-field representation of the bias-field
  NEWIMAGE::volume<float> GetField(double ksp, double lambda) const;
  /// Caclulate and return At matrix for debug purposes
  MISCMATHS::SpMat<float> GetAtMatrix(const EDDY::ImageCoordinates&  coords, 
				      const NEWIMAGE::volume<float>& predicted, 
				      const NEWIMAGE::volume<float>& mask) const;
  /// Calculate and return At* matrix for debug purposes
  MISCMATHS::SpMat<float> GetAtStarMatrix(const EDDY::ImageCoordinates&  coords, const NEWIMAGE::volume<float>& mask) const;
  ///
  NEWIMAGE::volume<float> GetATimesField(const EDDY::ImageCoordinates&  coords, 
					 const NEWIMAGE::volume<float>& predicted, 
					 const NEWIMAGE::volume<float>& mask,
					 const NEWIMAGE::volume<float>& field); 
  /// Write out current state for debug purposes
  void Write(const std::string& basename) const;
private:
  BiasFieldEstimatorImpl* _pimpl;
};

} // End namespace EDDY
#endif // End #ifndef BiasFieldEstimator_h
