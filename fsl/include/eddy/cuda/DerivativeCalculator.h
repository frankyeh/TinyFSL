/////////////////////////////////////////////////////////////////////
///
/// \file DerivativeCalculator.h
/// \brief Declarations of class used to calculate the derivatives of a prediction in scan space w.r.t. all parameters.
///
/// \author Jesper Andersson
/// \version 1.0b, Dec., 2019.
/// \Copyright (C) 2012 University of Oxford
///
/////////////////////////////////////////////////////////////////////

#ifndef DerivativeCalculator_h
#define DerivativeCalculator_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <thrust/system_error.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#pragma push
#pragma diag_suppress = code_is_unreachable // Supress warnings from armawrap
#pragma diag_suppress = expr_has_no_effect  // Supress warnings from boost
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#pragma pop
#include "EddyHelperClasses.h"
#include "ECScanClasses.h"
#include "EddyKernels.h"
#include "CudaVolume.h"
#include "EddyInternalGpuUtils.h"

namespace EDDY {

enum class DerivType { Old, Mixed };

/****************************************************************//**
*
*  \brief Calculates derivatives for use in EddyInternalUtils::param_update
*
********************************************************************/
class DerivativeCalculator
{
public:
  /// Constructor that calculates all the derivatives in scan space
  DerivativeCalculator(CudaVolume&        pred,                /// [in] Prediction in model space
		       CudaVolume&        pmask,               /// [in] Predefined mask in model space
		       ECScan&            scan,                /// [in] Scan
		       const CudaVolume&  susc,                /// [in] Susceptibility field
		       Parameters         whichp,              /// [in] Specifies whis parameters to calculate derivatives for
		       float              fwhm,                /// [in] FWHM for optional smoothing of derivative images
		       DerivType          dt=DerivType::Old)   /// [in] Specify details of how to calculate derivatives
  EddyTry : _dt(dt), _fwhm(fwhm), _whichp(whichp), _derivs(pred,scan.NDerivs(whichp),false), _dfield(pred,3,false), _pios(pred,false), _mios(pmask,false), _jac(pred,false)
  {
    if (dt == DerivType::Old) calculate_direct_derivatives(pred,pmask,scan,susc,whichp);
    else if (dt == DerivType::Mixed) calculate_mixed_derivatives(pred,pmask,scan,susc,whichp);
    else throw EDDY::EddyException("DerivativeCalculator::DerivativeCalculator: Unknown derivative type");

    if (fwhm) _derivs.Smooth(fwhm,_mios);
  } EddyCatch
  /// Returns prediction in scan space
  const CudaVolume& PredInScanSpace() const { return(_pios); }
  /// Returns mask in scan space
  const CudaVolume& MaskInScanSpace() const { return(_mios); }
  /// Returns Jacobian determinant map in scan space
  const CudaVolume& JacInScanSpace() const { return(_jac); }
  /// Returns a const reference to the pre-calculated derivatives
  const CudaVolume4D& Derivatives() const { return(_derivs); }
  /// Returns a value indicating how the derivatives were calculated.
  DerivType WhichDerivativeType() const { return(_dt); }
  /// Writes derivatives as 4D nifti and other images as 3D niftis
  void Write(const std::string& basename) const;
private:
  DerivType                  _dt;     /// Flag that indicates how derivatives were calculated
  float                      _fwhm;   /// FWHM of optional smoothing of derivative images
  Parameters                 _whichp; /// Specifies whis parameters to calculate derivatives for
  CudaVolume4D               _derivs; /// The partial derivative images
  CudaVolume4D               _dfield; /// The displacement field for (original) model->scan transformation
  CudaVolume                 _pios;   /// Prediction in scan (original) space
  CudaVolume                 _mios;   /// Indicates where pred in scan space is valid
  CudaVolume                 _jac;    /// Jacobian in scan space
  /// Calculates partial derivatives using finite differences and interpolation given by scan
  void calculate_direct_derivatives(CudaVolume& pred, CudaVolume& pmask, ECScan& scan, const CudaVolume& susc, Parameters whichp);
  /// Experimental
  void calculate_mixed_derivatives(CudaVolume& pred, CudaVolume& pmask, ECScan& scan, const CudaVolume& susc, Parameters whichp);
  /// Caclulates field for model-to-scan transform given parameters in scan
  void get_field(const EDDY::ECScan&         scan,
		 const EDDY::CudaVolume&     susc,
		 const EDDY::CudaVolume4D&   infield,
		 EDDY::CudaVolume&           mask,
		 EDDY::CudaVolume4D&         field,
		 EDDY::CudaVolume&           jac) const;
  /// Transform from model to scan space, give a model-to-scan-field as input.
  void transform_to_scan_space(const EDDY::CudaVolume&       pred,
			       const EDDY::ECScan&           scan,
			       const EDDY::CudaVolume4D&     dfield,
			       EDDY::CudaVolume&             oima,
			       EDDY::CudaVolume&             omask) const;
  /// Inverts the field.
  void invert_field(const CudaVolume4D&               field,
		    const EDDY::AcqPara&              acqp,
		    const CudaVolume&                 inmask,
		    CudaVolume4D&                     ifield,
		    CudaVolume&                       omask) const;

  /// Inverts the field, using the inifield for bracketing the new inverse
  void invert_field(const CudaVolume4D&               field,
		    const EDDY::AcqPara&              acqp,
		    const CudaVolume&                 inmask,
		    const CudaVolume4D&               inifield,
		    CudaVolume4D&                     ifield) const;

  void voxel_2_mm_displacements(CudaVolume4D& field, unsigned int dir) const;

  void mm_2_voxel_displacements(CudaVolume4D& field, unsigned int dir) const;

  void get_slice_modulated_deriv(// Input/Output
				 CudaVolume4D&              derivs,
				 // Input
				 const CudaVolume&          mask,
				 unsigned int               primi,
				 unsigned int               scndi,
				 const SliceDerivModulator& sdm) const;

};

/****************************************************************//**
*
*  \fn DerivativeCalculator::DerivativeCaclulator(CudaVolume&        pred,
*		                                  CudaVolume&        pmask,
*		                                  ECScan&            scan,
*		                                  const CudaVolume&  susc,
*		                                  Parameters         whichp,
*		                                  bool               fast=false)
*  \brief Constructor for the DerivativeCalculator class
*
*
********************************************************************/

} // End namespace EDDY

#endif // End #ifndef DerivativeCalculator_h

// Dead code
/*
/// Calculates partial derivatives using finite differences and tri-linear interpolation
void calculate_direct_derivatives_fast(CudaVolume& pred, CudaVolume& pmask, ECScan& scan, const CudaVolume& susc, Parameters whichp);
/// Experimental
void calculate_direct_derivatives_very_fast(CudaVolume& pred, CudaVolume& pmask, ECScan& scan, const CudaVolume& susc, Parameters whichp);
/// Experimental
void calculate_modulated_derivatives(CudaVolume& pred, CudaVolume& pmask, ECScan& scan, const CudaVolume& susc, Parameters whichp);

/// Finds the two bracketing indicies for each point of the inverse field, and returns the lower index
void get_lower_bound_indicies(const CudaVolume4D&         field,
                              const EDDY::AcqPara&        acqp,
                              const CudaVolume&           inmask,
			      thrust::device_vector<int>& lbindx,
			      CudaVolume&                 omask) const;

void get_spatially_modulated_deriv(// Input/Output
				   CudaVolume4D&                derivs,
				   // Input
				   const CudaVolume&            mask,
				   unsigned int                 primi,
				   unsigned int                 scndi,
				   const SpatialDerivModulator& sdm,
				   const CudaVolume&            offset) const;
*/
