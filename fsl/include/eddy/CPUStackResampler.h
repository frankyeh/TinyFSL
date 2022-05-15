/*! \file CPUStackResampler.h
    \brief Contains declaration of a class for spline/tri-linear resampling of irregularly sampled columns in the z-direction.

    \author Jesper Andersson
    \version 1.0b, May, 2021.
*/
//
// CPUStackResampler.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2021 University of Oxford
//

#ifndef CPUStackResampler_h
#define CPUStackResampler_h

#include <cstdlib>
#include <vector>
#include <armadillo>
#include "newimage/newimageall.h"
#include "EddyHelperClasses.h"

namespace EDDY {

class CPUStackResampler
{
public:
  /// Constructor. Performs the actual work so that when the object is created there is already a resampled image ready.

  /// This version of the constructor uses a predicted volume and the laplacian for regularisation
  CPUStackResampler(const NEWIMAGE::volume<float>&  stack,
		    const NEWIMAGE::volume<float>&  zcoord,
		    const NEWIMAGE::volume<float>&  pred,
		    const NEWIMAGE::volume<float>&  mask,
		    double                          lambda=0.005);

  /// This version of the constructor uses either splines and Laplacian regularisation or linear interpolation.
  CPUStackResampler(const NEWIMAGE::volume<float>&  stack,
		    const NEWIMAGE::volume<float>&  zcoord,
		    const NEWIMAGE::volume<float>&  mask,
		    NEWIMAGE::interpolation         interp=NEWIMAGE::spline,
		    double                          lambda=0.005);

  ~CPUStackResampler() {}

  // Returns interpolated image
  const NEWIMAGE::volume<float>& GetImage() const EddyTry { return(_ovol); } EddyCatch
  // Returns mask
  const NEWIMAGE::volume<float>& GetMask() const EddyTry { return(_omask); } EddyCatch

private:
  NEWIMAGE::volume<float> _ovol;
  NEWIMAGE::volume<float> _omask;
  void spline_interpolate_slice_stack(// Input
				      const NEWIMAGE::volume<float>&   slice_stack,
				      const NEWIMAGE::volume<float>&   z_coord,
				      const NEWIMAGE::volume<float>&   stack_mask,
				      double                           lambda,
				      // Optional input
				      const NEWIMAGE::volume<float>    *pred_ptr,
				      // Output
				      NEWIMAGE::volume<float>&         ovol,
				      NEWIMAGE::volume<float>&         omask);
  void linear_interpolate_slice_stack(// Input
				      const NEWIMAGE::volume<float>&   slice_stack,
				      const NEWIMAGE::volume<float>&   z_coord,
				      const NEWIMAGE::volume<float>&   stack_mask,
				      // Output
				      NEWIMAGE::volume<float>&         ovol,
				      NEWIMAGE::volume<float>&         omask);
  arma::Mat<float> get_StS(int sz, float lambda) const;
  arma::Mat<float> get_regular_W(int sz) const;
  arma::Mat<float> get_Wir(const NEWIMAGE::volume<float>& zcoord,
					    int i, int j) const;
  arma::Col<float> get_y(const NEWIMAGE::volume<float>& stack,
			 int i, int j) const EddyTry {
    arma::Col<float> y(stack.zsize());
    for (int k=0; k<stack.zsize(); k++) y[k] = stack(i,j,k);
    return(y);
  } EddyCatch
  float wgt_at(float x) const;
  std::vector<float> sort_zcoord(const NEWIMAGE::volume<float>& zcoord,
				 int i, int j) const;
  arma::Mat<float> get_prediction_weights(const std::vector<float> zcoord) const;
};

} // End namespace EDDY

#endif // End #ifndef CPUStackResampler_h
