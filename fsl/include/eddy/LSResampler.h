/*! \file LSResampler.h
    \brief Contains declaration of class for least-squares resampling of pairs of images

    \author Jesper Andersson
    \version 1.0b, April, 2013.
*/
// 
// LSResampler.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford 
//

#ifndef LSResampler_h
#define LSResampler_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <time.h>
#include "newimage/newimageall.h"
#include "ECScanClasses.h"

namespace EDDY {

class LSResamplerImpl;

/****************************************************************//**
*
* \brief Class used to perform Least-Squares resampling of a pair
* of images acquired with reversed PE-blips.
*
* Class used to perform Least-Squares resampling of a pair
* of images acquired with reversed PE-blips. It is implemented
* using the "Pimpl idiom" which means that this class only
* implements an interface whereas the actual work is being performed
* by the LSResamplerImpl class which is declared and defined in
* LSResampler.cpp or cuda/LSResampler.cu depending on what platform
* the code is compiled for.
*
* The way you would use it is by first constructing the object and
* then calling `GetResampledVolume()` to retrieve the resampled image
* and, possibly, `GetMask()` to get a binary mask indicating valid voxels.
*
* Example:
* ~~~{.c}
* LSResampler my_resampler(blipup,blipdown,field);
* NEWIMAGE::volume<float> my_hifi_image = my_resampler.GetResampledVolume();
* ~~~
********************************************************************/ 
class LSResampler
{
public:
  /// Constructor. Performs the actual work so that when the object is created there is already a resampled image ready.
  LSResampler(const EDDY::ECScan&                               s1, 
	      const EDDY::ECScan&                               s2,
	      std::shared_ptr<const NEWIMAGE::volume<float> >   hzfield,
	      double                                            lambda=0.01);
  ~LSResampler();
  /// Returns the resampled image. N.B. one resampled image per pair of input images.
  const NEWIMAGE::volume<float>& GetResampledVolume() const;
  /// Returns a binary mask with one for the voxels where a valid resampling could be performed.
  const NEWIMAGE::volume<float>& GetMask() const;
private:
  LSResamplerImpl* _pimpl;
};

} // End namespace EDDY
#endif // End #ifndef LSResampler_h
