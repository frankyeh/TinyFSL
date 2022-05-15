/////////////////////////////////////////////////////////////////////
///
/// \file DerivativeCalculator.cu
/// \brief Definitions of class used to calculate the derivatives of a prediction in scan space w.r.t. all parameters.
///
/// \author Jesper Andersson
/// \version 1.0b, Dec., 2019.
/// \Copyright (C) 2012 University of Oxford
///
/////////////////////////////////////////////////////////////////////

// Because of a bug in cuda_fp16.hpp, that gets included by cublas_v2.h, it has to
// be included before any include files that set up anything related to the std-lib.
// If not, there will be an ambiguity in cuda_fp16.hpp about wether to use the
// old-style C isinf or the new (since C++11) std::isinf.
#include "cublas_v2.h"

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <ctime>
#include <cuda.h>
#include <thrust/system_error.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/inner_product.h>
#pragma push
#pragma diag_suppress = code_is_unreachable // Supress warnings from armawrap
#pragma diag_suppress = expr_has_no_effect  // Supress warnings from boost
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#pragma pop
#include "utils/FSLProfiler.h"
#include "EddyHelperClasses.h"
#include "EddyUtils.h"
#include "EddyCudaHelperFunctions.h"
#include "EddyGpuUtils.h"
#include "EddyKernels.h"
#include "EddyFunctors.h"
#include "DerivativeCalculator.h"
using namespace EDDY;


/****************************************************************//**
*
*  Writes out information useful mainly for debugging purposes.
*
*  \param[in] basename Common "basename" for all output files.
*
********************************************************************/
void DerivativeCalculator::Write(const std::string& basename) const EddyTry
{
  _derivs.Write(basename+"_derivatives_in_scan_space");
  _pios.Write(basename+"_prediction_in_scan_space");
  _mios.Write(basename+"_mask_in_scan_space");
  _jac.Write(basename+"_jacobian_in_scan_space");
  return;
} EddyCatch

/****************************************************************//**
*
*  Calculates the derivatives of the prediction pred with respect
*  to the parameters given by whichp. The derivatives are in the
*  scan space. This call will use only functions/kernels that yield
*  results identical to the "original" implementation.
*
*  \param[in] pred Prediction in model space
*  \param[in] mask Predefined mask in model space
*  \param[in] scan Scan that we want derivatives for
*  \param[in] susc Susceptibility field
*  \param[in] whichp Specifies which parameters we want the
*  derivatives with respect to
*
********************************************************************/
void DerivativeCalculator::calculate_direct_derivatives(CudaVolume&       pred,
							CudaVolume&       pmask,
							ECScan&           scan,
							const CudaVolume& susc,
							Parameters        whichp) EddyTry
{
  static Utilities::FSLProfiler prof("_"+std::string(__FILE__)+"_"+std::string(__func__));
  double total_key = prof.StartEntry("Total");
  // Check input parameters
  if (pred != scan.GetIma()) throw EDDY::EddyException("DerivativeCalculator::calculate_direct_derivatives: pred-scan size mismatch");
  if (susc.Size() && susc != scan.GetIma()) throw EDDY::EddyException("DerivativeCalculator::calculate_direct_derivatives: susc-scan size mismatch");

  EDDY::CudaVolume mask(pred,false);
  EDDY::CudaVolume4D grad;     // Zero size placeholder
  // Calculated prediction in scan space. Also serves as base for derivative calculations.
  EddyInternalGpuUtils::transform_model_to_scan_space(pred,scan,susc,true,_pios,mask,_jac,grad);
  // Transform predefined mask to scan space and combine with sampling mask
  EDDY::CudaVolume skrutt;     // Zero size placeholder
  EDDY::CudaVolume4D skrutt4D; // Zero size placeholder
  EddyInternalGpuUtils::transform_model_to_scan_space(pmask,scan,susc,false,_mios,skrutt,skrutt,skrutt4D);
  // Binarise resampled prediction mask and combine with sampling mask
  _mios.Binarise(0.99); _mios *= mask;
  // Calculate derivatives
  EDDY::CudaVolume perturbed(pred,false);
  EDDY::CudaVolume jac(pred,false);
  EDDY::ECScan sc = scan;
  for (unsigned int i=0; i<sc.NDerivs(whichp); i++) {
    double p = sc.GetDerivParam(i,whichp);
    sc.SetDerivParam(i,p+sc.GetDerivScale(i,whichp),whichp);
    EddyInternalGpuUtils::transform_model_to_scan_space(pred,sc,susc,true,perturbed,mask,jac,grad);
    _derivs[i] = _mios * (perturbed-_pios)/sc.GetDerivScale(i,whichp);
    sc.SetDerivParam(i,p,whichp);
  }
  prof.EndEntry(total_key);
  return;
} EddyCatch

/****************************************************************//**
*
*  Calculates the derivatives of the prediction pred with respect
*  to the parameters given by whichp. It uses modulation for the
*  movement parameters in the case of slice-to-vol, but direct
*  calculation for the other parameters.
*
*  \param[in] pred Prediction in model space
*  \param[in] scan Scan that we want derivatives for
*  \param[in] susc Susceptibility field
*  \param[in] whichp Specifies which parameters we want the
*  derivatives with respect to
*
********************************************************************/
void DerivativeCalculator::calculate_mixed_derivatives(CudaVolume&       pred,
						       CudaVolume&       pmask,
						       ECScan&           scan,
						       const CudaVolume& susc,
						       Parameters        whichp) EddyTry
{
  // Check input parameters
  if (pred != scan.GetIma()) throw EDDY::EddyException("DerivativeCalculator::calculate_mixed_derivatives: pred-scan size mismatch");
  if (susc.Size() && susc != scan.GetIma()) throw EDDY::EddyException("DerivativeCalculator::calculate_mixed_derivatives: susc-scan size mismatch");

  EDDY::CudaVolume mask(pred,false);  // Total mask
  EDDY::CudaVolume fmask(pred,false); // Mask where field is valid
  EDDY::CudaVolume4D skrutt4D;        // Zero size placeholder
  // Get field for model->scan_space transform
  this->get_field(scan,susc,skrutt4D,fmask,_dfield,_jac);
  // Calculate prediction in scan space. Also serves as base for derivative calculations.
  this->transform_to_scan_space(pred,scan,_dfield,_pios,mask);
  _pios *= _jac;
  // Transform predefined mask to scan space and combine with sampling mask
  EDDY::CudaVolume skrutt;     // Zero size placeholder
  this->transform_to_scan_space(pmask,scan,_dfield,_mios,skrutt);
  // Binarise resampled prediction mask and combine with sampling mask
  _mios.Binarise(0.99); _mios *= mask; _mios *= fmask;

  EDDY::CudaVolume4D pdfield(pred,3,false);
  EDDY::CudaVolume jac(pred,false);   // Jacobian determinant of field
  EDDY::CudaVolume perturbed(pred,false);
  // We are relying on the order of derivatives being movement followed by EC.
  // First we calculate the movement derivatives using modulation.
  if (whichp == EDDY::ALL || whichp == EDDY::MOVEMENT) { // If we are asked for movement
    for (unsigned int i=0; i<scan.NCompoundDerivs(EDDY::MOVEMENT); i++) {
      // First calculate primary derivative for the compound
      EDDY::DerivativeInstructions di = scan.GetCompoundDerivInstructions(i,EDDY::MOVEMENT);
      double p = scan.GetDerivParam(di.GetPrimaryIndex(),EDDY::MOVEMENT);
      scan.SetDerivParam(di.GetPrimaryIndex(),p+di.GetPrimaryScale(),EDDY::MOVEMENT);
      this->get_field(scan,susc,_dfield,fmask,pdfield,jac);
      this->transform_to_scan_space(pred,scan,pdfield,perturbed,skrutt);
      perturbed *= jac;
      _derivs[di.GetPrimaryIndex()] = _mios * (perturbed-_pios) / di.GetPrimaryScale();
      scan.SetDerivParam(di.GetPrimaryIndex(),p,EDDY::MOVEMENT);
      // Next calculate any secondary/modulated derivatives
      if (di.IsSliceMod()) {
	for (unsigned int j=0; j<di.NSecondary(); j++) {
	  EDDY::SliceDerivModulator sdm = di.GetSliceModulator(j);
	  get_slice_modulated_deriv(_derivs,mask,di.GetPrimaryIndex(),di.GetSecondaryIndex(j),sdm);
	}
      }
      else if (di.IsSpatiallyMod()) throw EDDY::EddyException("DerivativeCalculator::calculate_mixed_derivatives: Spatial modulation requested");
    }
  }
  // Next we calculate the EC derivatives using "direct derivatives"
  if (whichp == EDDY::ALL || whichp == EDDY::EC) { // If we are asked for EC (eddy currents)
    unsigned int offset = (whichp == EDDY::ALL) ? scan.NDerivs(EDDY::MOVEMENT) : 0;
    for (unsigned int i=0; i<scan.NDerivs(EDDY::EC); i++) {
      double p = scan.GetDerivParam(i,EDDY::EC);
      scan.SetDerivParam(i,p+scan.GetDerivScale(i,EDDY::EC),EDDY::EC);
      this->get_field(scan,susc,_dfield,fmask,pdfield,jac);
      this->transform_to_scan_space(pred,scan,pdfield,perturbed,skrutt);
      perturbed *= jac;
      _derivs[offset+i] = _mios * (perturbed-_pios) / scan.GetDerivScale(i,EDDY::EC);
      scan.SetDerivParam(i,p,EDDY::EC);
    }
  }

  return;
} EddyCatch

/****************************************************************//**
*
*  Calculates the field for model-to-scan transformation.
*
*  \param[in] scan Scan that we want derivatives for
*  \param[in] susc Susceptibility field
*  \param[in] infield Displacement field calculated by an earlier
*  call. If it has size zero it will be ignored. If size is non-zero it
*  will be passed into kernels calculating the inverse.
*  \param[in,out] mask If infield has zero size a mask is calculated and
*  returned in mask. If infield is non-zero it is expected that mask is
*  mask specifying where the field is valid.
*  \param[out] field The calculated displacement field
*  \param[out] mask Mask with non-zero values where the field is valid.
*  \param[out] jac Jacobian determinants of field
*
********************************************************************/
void DerivativeCalculator::get_field(// Input
				     const EDDY::ECScan&            scan,
				     const EDDY::CudaVolume&        susc,
				     const EDDY::CudaVolume4D&      infield,
				     // Input/output
				     EDDY::CudaVolume&              mask,
				     // Output
				     EDDY::CudaVolume4D&            field,
				     EDDY::CudaVolume&              jac) const EddyTry
{
  EDDY::CudaVolume tot(scan.GetIma(),false);              // Total (EC and susc) field
  EDDY::CudaVolume smask(scan.GetIma(),false); smask = 1.0; // Defines where transformed susc field is valid
  unsigned int dir = (scan.GetAcqPara().PhaseEncodeVector()(1)!=0) ? 0 : 1;
  // Get EC field and combine with susc field
  EddyInternalGpuUtils::get_ec_field(scan,tot);
  if (susc.Size()) {
    EDDY::CudaVolume tsusc(susc,false);
    if (scan.IsSliceToVol()) {
      std::vector<NEWMAT::Matrix> R = EddyUtils::GetSliceWiseForwardMovementMatrices(scan);
      EDDY::CudaVolume4D skrutt;
      EddyInternalGpuUtils::affine_transform(susc,R,tsusc,skrutt,smask);
    }
    else {
      NEWMAT::Matrix R = scan.ForwardMovementMatrix();
      EDDY::CudaVolume4D skrutt;
      EddyInternalGpuUtils::affine_transform(susc,R,tsusc,skrutt,smask);
    }
    tot += tsusc;
  }
  // Convert Hz map to voxel displacement field
  EDDY::CudaVolume4D dfield(tot,3,false);
  FieldGpuUtils::Hz2VoxelDisplacements(tot,scan.GetAcqPara(),dfield);
  if (infield.Size()) {
    field = infield;
    this->mm_2_voxel_displacements(field,dir);
    this->invert_field(dfield,scan.GetAcqPara(),mask,field,field);
  }
  else {
    this->invert_field(dfield,scan.GetAcqPara(),smask,field,mask);
  }
  // Get Jacobian of inverted field
  if (jac.Size()) field.SampleTrilinearDerivOnVoxelCentres(dir,mask,jac,true);
  // Convert voxel displacement field to mm
  this->voxel_2_mm_displacements(field,dir);

  return;
} EddyCatch

void DerivativeCalculator::transform_to_scan_space(// Input
						   const EDDY::CudaVolume&       pred,
						   const EDDY::ECScan&           scan,
						   const EDDY::CudaVolume4D&     dfield,
						   // Output
						   EDDY::CudaVolume&             oima,
						   EDDY::CudaVolume&             omask) const EddyTry
{
  // Some input checking
  if (oima != pred) oima.SetHdr(pred);
  if (omask != pred) omask.SetHdr(pred);
  EDDY::CudaVolume4D grad; // Zero size place holder
  NEWMAT::IdentityMatrix I(4);
  if (scan.IsSliceToVol()) {
    // Get RB matrices, one per slice.
    std::vector<NEWMAT::Matrix> R = EddyUtils::GetSliceWiseForwardMovementMatrices(scan);
    std::vector<NEWMAT::Matrix> II(R.size()); for (unsigned int i=0; i<R.size(); i++) II[i] = I;
    // Transform prediction/model
    EddyInternalGpuUtils::general_transform(pred,II,dfield,R,oima,grad,omask);
  }
  else {
    // Get RB matrix
    NEWMAT::Matrix R = scan.ForwardMovementMatrix();
    // Transform prediction/model
    EddyInternalGpuUtils::general_transform(pred,I,dfield,R,oima,grad,omask);
  }

  return;
} EddyCatch

void DerivativeCalculator::invert_field(// Input
                                        const CudaVolume4D&               field,
					const EDDY::AcqPara&              acqp,
					const CudaVolume&                 inmask,
					// Output
					CudaVolume4D&                     ifield,
					CudaVolume&                       omask) const EddyTry
{
  int tpb = field.Size(0);
  int nblocks = field.Size(2);
  if (acqp.PhaseEncodeVector()(1) != 0.0) { // If PE in x
    EddyKernels::invert_displacement_field_x<<<nblocks,tpb>>>(field.GetPtr(0),inmask.GetPtr(),field.Size(0),field.Size(1),
							      ifield.GetPtr(0),omask.GetPtr(),nblocks*tpb);
    EddyCudaHelperFunctions::CudaSync("EddyKernels::invert_displacement_field_x");
  }
  else if (acqp.PhaseEncodeVector()(2) != 0.0) { // If PE in y
    EddyKernels::invert_displacement_field_y<<<nblocks,tpb>>>(field.GetPtr(1),inmask.GetPtr(),field.Size(0),field.Size(1),
							      ifield.GetPtr(1),omask.GetPtr(),nblocks*tpb);
    EddyCudaHelperFunctions::CudaSync("EddyKernels::invert_displacement_field_y");
  }
  else throw EddyException("DerivativeCaclulator::invert_field_1: Invalid phase encode vector");
  return;
} EddyCatch

void DerivativeCalculator::invert_field(// Input
                                        const CudaVolume4D&               field,
					const EDDY::AcqPara&              acqp,
					const CudaVolume&                 inmask,
					const CudaVolume4D&               inifield,
					// Output
					CudaVolume4D&                     ifield) const EddyTry
{
  int tpb = field.Size(0);
  int nblocks = field.Size(2);
  if (acqp.PhaseEncodeVector()(1) != 0.0) { // If PE in x
    EddyKernels::invert_displacement_field_x<<<nblocks,tpb>>>(field.GetPtr(0),inmask.GetPtr(),inifield.GetPtr(0),
							      field.Size(0),field.Size(1),ifield.GetPtr(0),nblocks*tpb);
    EddyCudaHelperFunctions::CudaSync("EddyKernels::invert_displacement_field_x");
  }
  else if (acqp.PhaseEncodeVector()(2) != 0.0) { // If PE in y
    EddyKernels::invert_displacement_field_y<<<nblocks,tpb>>>(field.GetPtr(1),inmask.GetPtr(),inifield.GetPtr(1),
							      field.Size(0),field.Size(1),ifield.GetPtr(1),nblocks*tpb);
    EddyCudaHelperFunctions::CudaSync("EddyKernels::invert_displacement_field_y");
  }
  else throw EddyException("DerivativeCaclulator::invert_field_2: Invalid phase encode vector");
  return;
} EddyCatch

void DerivativeCalculator::voxel_2_mm_displacements(// Input/Output
						    CudaVolume4D&  field,
						    // Input
						    unsigned int   dir) const EddyTry
{
  thrust::transform(field.Begin(dir),field.End(dir),field.Begin(dir),EDDY::MulByScalar<float>(field.Vxs(dir)));
  return;
} EddyCatch

void DerivativeCalculator::mm_2_voxel_displacements(// Input/Output
						    CudaVolume4D&  field,
						    // Input
						    unsigned int   dir) const EddyTry
{
  thrust::transform(field.Begin(dir),field.End(dir),field.Begin(dir),EDDY::MulByScalar<float>(1.0/field.Vxs(dir)));
  return;
} EddyCatch

void DerivativeCalculator::get_slice_modulated_deriv(// Input/Output
						     CudaVolume4D&              derivs,
						     // Input
						     const CudaVolume&          mask,
						     unsigned int               primi,
						     unsigned int               scndi,
						     const SliceDerivModulator& sdm) const EddyTry
{
  thrust::device_vector<float> dmod = sdm.GetMod();
  int tpb = derivs.Size(0);
  int nblocks = derivs.Size(2);

  EddyKernels::slice_modulate_deriv<<<nblocks,tpb>>>(derivs.GetPtr(primi),mask.GetPtr(),derivs.Size(0),derivs.Size(1),derivs.Size(2),
                                                     thrust::raw_pointer_cast(dmod.data()),derivs.GetPtr(scndi),tpb*nblocks);
  EddyCudaHelperFunctions::CudaSync("DerivativeCalculator::get_slice_modulated_deriv::slice_modulate_deriv");
  return;
} EddyCatch



// This section has some dead code that may (but probably not) be
// useful in the future.

  /*
  auto start_invert = std::chrono::system_clock::now();
  FieldGpuUtils::InvertDisplacementField(dfield,scan.GetAcqPara(),mask,idfield,omask);
  auto end_invert = std::chrono::system_clock::now();
  // Get jacobian of inverted field
  auto start_jacobian = std::chrono::system_clock::now();
  if (jac.Size()) FieldGpuUtils::GetJacobian(idfield,scan.GetAcqPara(),jac);
  auto end_jacobian = std::chrono::system_clock::now();
  CudaVolume new_jac(jac,false);
  auto start_new_jacobian = std::chrono::system_clock::now();
  auto end_new_jacobian = std::chrono::system_clock::now();
  // Transform field to mm displacements
  FieldGpuUtils::Voxel2MMDisplacements(idfield);

  auto end = std::chrono::system_clock::now();

  char fname[256];
  sprintf(fname,"old_jac_%03d",cnt);
  NEWIMAGE::write_volume(jac.GetVolume(),fname);
  sprintf(fname,"new_jac_%03d",cnt);
  NEWIMAGE::write_volume(new_jac.GetVolume(),fname);

  std::chrono::duration<double> duration = end-start;
  std::chrono::duration<double> inv_duration = end_invert-start_invert;
  std::chrono::duration<double> jac_duration = end_jacobian-start_jacobian;
  std::chrono::duration<double> new_jac_duration = end_new_jacobian-start_new_jacobian;
  std::chrono::duration<double> duration1 = start_second_part - start;
  std::chrono::duration<double> duration2 = end - start_second_part;


  cout << "EddyInternalGpuUtils::field_for_model_to_scan_transform took " << duration.count() << " sec, of which the inverse was " << 100.0*inv_duration.count()/duration.count() << " %, and the Jacobian was " << 100.0*jac_duration.count()/duration.count() << " %" << endl;
  cout << "EddyInternalGpuUtils::field_for_model_to_scan_transform took " << duration.count() << " sec, of which the first half was " << 100.0*duration1.count()/duration.count() << " %, and the second half " << 100.0*duration2.count()/duration.count() << " %" << endl;
  cout << "EddyInternalGpuUtils::field_for_model_to_scan_transform old Jacobian took " << jac_duration.count() << " sec, and new Jacobian took " << new_jac_duration.count() << " sec" << endl;
*/

/*

void DerivativeCalculator::calculate_direct_derivatives_very_fast(CudaVolume&       pred,
								  CudaVolume&       pmask,
								  ECScan&           scan,
								  const CudaVolume& susc,
								  Parameters        whichp) EddyTry
{
  // Check input parameters
  if (pred != scan.GetIma()) throw EDDY::EddyException("DerivativeCalculator::calculate_direct_derivatives_very_fast: pred-scan size mismatch");
  if (susc.Size() && susc != scan.GetIma()) throw EDDY::EddyException("DerivativeCalculator::calculate_direct_derivatives_very_fast: susc-scan size mismatch");

  EDDY::CudaVolume4D dfield(pred,3,false); dfield = 0.0;
  EDDY::CudaVolume fmask(pred,false); // Mask where field is valid
  EDDY::CudaVolume mask(pred,false);  // Mask where resampled image is valid
  EDDY::CudaVolume base(pred,false);  // zero-point for derivatives
  thrust::device_vector<int> lbindx;  // Lower bounds of range when calculating inverses

  this->get_field(scan,susc,lbindx,dfield,fmask,_jac);
  this->transform_to_scan_space(pred,scan,dfield,_pios,mask);
  _pios *= _jac; // Used for returning resampled predictions
  NEWIMAGE::interpolation old_interp = pred.Interp();
  if (old_interp != NEWIMAGE::trilinear) {
    pred.SetInterp(NEWIMAGE::trilinear);
    this->transform_to_scan_space(pred,scan,dfield,base,mask);
  }
  else base = _pios;
  // Transform predefined mask to scan space and combine with sampling mask
  this->transform_to_scan_space(pmask,scan,dfield,_mios,mask);
  // Binarise resampled prediction mask and combine with sampling mask
  _mios.Binarise(0.99); _mios *= mask;
  // char fname[256]; sprintf(fname,"field_%03d",1); dfield.Write(fname);

  EDDY::CudaVolume jac(pred,false);   // Jacobian determinant of field
  EDDY::CudaVolume perturbed(pred,false);
  for (unsigned int i=0; i<scan.NCompoundDerivs(whichp); i++) {
    // First calculate primary derivative for the compound
    EDDY::DerivativeInstructions di = scan.GetCompoundDerivInstructions(i,whichp);
    double p = scan.GetDerivParam(di.GetPrimaryIndex(),whichp);
    scan.SetDerivParam(di.GetPrimaryIndex(),p+di.GetPrimaryScale(),whichp);
    this->get_field(scan,susc,lbindx,dfield,fmask,jac);
    this->transform_to_scan_space(pred,scan,dfield,perturbed,mask);
    perturbed *= jac;
    _derivs[di.GetPrimaryIndex()] = (perturbed-base)/di.GetPrimaryScale();
    scan.SetDerivParam(di.GetPrimaryIndex(),p,whichp);
    // Next calculate any secondary/modulated derivatives
    for (unsigned int j=0; j<di.NSecondary(); j++) {
      if (di.IsSliceMod()) {
	EDDY::SliceDerivModulator sdm = di.GetSliceModulator(j);
	get_slice_modulated_deriv(_derivs,fmask,di.GetPrimaryIndex(),di.GetSecondaryIndex(j),sdm);
      }
      else if (di.IsSpatiallyMod()) {
	EDDY::SpatialDerivModulator sdm = di.GetSpatialModulator(j);
	// get_spatially_modulated_deriv(_derivs,fmask,di.GetPrimaryIndex(),di.GetSecondaryIndex(j),sdm);
      }
    }
  }

  if (old_interp != NEWIMAGE::trilinear) pred.SetInterp(old_interp);

  return;
} EddyCatch

void DerivativeCalculator::calculate_direct_derivatives_fast(CudaVolume&       pred,
							     CudaVolume&       pmask,
							     ECScan&           scan,
							     const CudaVolume& susc,
							     Parameters        whichp) EddyTry
{
  // Check input parameters
  if (pred != scan.GetIma()) throw EDDY::EddyException("DerivativeCalculator::calculate_direct_derivatives_fast: pred-scan size mismatch");
  if (susc.Size() && susc != scan.GetIma()) throw EDDY::EddyException("DerivativeCalculator::calculate_direct_derivatives_fast: susc-scan size mismatch");

  EDDY::CudaVolume4D skrutt;
  EDDY::CudaVolume fmask(pred,false); // Mask where field is valid
  EDDY::CudaVolume mask(pred,false);  // Mask where resampled image is valid

  this->get_field(scan,susc,skrutt,_dfield,fmask,_jac);
  this->transform_to_scan_space(pred,scan,_dfield,_pios,mask);
  _pios *= _jac; // Used for returning resampled predictions

  // Transform predefined mask to scan space and combine with sampling mask
  this->transform_to_scan_space(pmask,scan,dfield,_mios,mask);
  // Binarise resampled prediction mask and combine with sampling mask
  _mios.Binarise(0.99); _mios *= mask;
  char fname[256]; sprintf(fname,"field_%03d",1); dfield.Write(fname);

  EDDY::CudaVolume4D dfield(pred,3,false); // Displacements of perturbed field
  EDDY::CudaVolume jac(pred,false);        // Jacobian determinant of perturbed field
  EDDY::CudaVolume perturbed(pred,false);
  for (unsigned int i=0; i<scan.NDerivs(whichp); i++) {
    double p = scan.GetDerivParam(i,whichp);
    scan.SetDerivParam(i,p+scan.GetDerivScale(i,whichp),whichp);
    // auto start_get_field = std::chrono::system_clock::now();
    this->get_field(scan,susc,_dfield,dfield,fmask,jac);
    // auto end_get_field = std::chrono::system_clock::now();
    // std::chrono::duration<double> duration = end_get_field-start_get_field;
    // cout << "this->get_field(scan,susc,lbindx,dfield,fmask,jac); took " << duration.count() << " seconds" << endl;
    sprintf(fname,"field_%03d",i+2); dfield.Write(fname);
    this->transform_to_scan_space(pred,scan,dfield,perturbed,mask);
    perturbed *= jac;
    sprintf(fname,"perturbed_%03d",i+1); perturbed.Write(fname);
    _derivs[i] = (perturbed-_pios)/scan.GetDerivScale(i,whichp);
    scan.SetDerivParam(i,p,whichp);
  }

  return;
} EddyCatch

void DerivativeCalculator::calculate_modulated_derivatives(CudaVolume&       pred,
							   CudaVolume&       pmask,
							   ECScan&           scan,
							   const CudaVolume& susc,
							   Parameters        whichp) EddyTry
{
  // Check input parameters
  if (pred != scan.GetIma()) throw EDDY::EddyException("DerivativeCalculator::calculate_modulated_derivatives: pred-scan size mismatch");
  if (susc.Size() && susc != scan.GetIma()) throw EDDY::EddyException("DerivativeCalculator::calculate_modulated_derivatives: susc-scan size mismatch");

  // Check for the case of EC estimation without a susceptibiltity field, which means that
  // the constant off-resonance field should not be estimated. That also means that one of
  // the derivatives that we need in order to estimate the other should be estimated, but
  // not be included with the other derivatives.
  EDDY::CudaVolume offset_deriv;
  if (scan.NDerivs() < scan.NParam()) { // Indicates that the constant off-resonance field is not directly estimated
    if (susc.Size()) throw EDDY::EddyException("DerivativeCalculator::calculate_modulated_derivatives: pred-scan size mismatch");
    offset_deriv.SetHdr(pred);
  }

  EDDY::CudaVolume mask(pred,false);
  EDDY::CudaVolume4D grad;     // Zero size placeholder
  // Calculate prediction in scan space. Also serves as base for derivative calculations.
  EddyInternalGpuUtils::transform_model_to_scan_space(pred,scan,susc,true,_pios,mask,_jac,grad);
  // Transform predefined mask to scan space and combine with sampling mask
  EDDY::CudaVolume skrutt;     // Zero size placeholder
  EDDY::CudaVolume4D skrutt4D; // Zero size placeholder
  EddyInternalGpuUtils::transform_model_to_scan_space(pmask,scan,susc,false,_mios,skrutt,skrutt,skrutt4D);
  // Binarise resampled prediction mask and combine with sampling mask
  _mios.Binarise(0.99); _mios *= mask;

  EDDY::CudaVolume jac(pred,false);   // Jacobian determinant of field
  EDDY::CudaVolume perturbed(pred,false);
  // cout << "scan.NCompoundDerivs(whichp) = " << scan.NCompoundDerivs(whichp) << endl; cout.flush();
  for (unsigned int i=0; i<scan.NCompoundDerivs(whichp); i++) {
    // cout << "i = " << i << endl; cout.flush();
    // First calculate primary derivative for the compound
    EDDY::DerivativeInstructions di = scan.GetCompoundDerivInstructions(i,whichp);
    // cout << "di.NSecondary() = " << di.NSecondary() << endl; cout.flush();
    if (offset_deriv.Size()) {
      double p = scan.GetDerivParam(di.GetPrimaryIndex(),whichp,true);
      scan.SetDerivParam(di.GetPrimaryIndex(),p+di.GetPrimaryScale(),whichp,true);
      EddyInternalGpuUtils::transform_model_to_scan_space(pred,scan,susc,true,perturbed,mask,jac,grad);
      if (di.GetPrimaryIndex() == scan.NDerivs()) offset_deriv = _mios * (perturbed-_pios) / di.GetPrimaryScale();
      else _derivs[di.GetPrimaryIndex()] = _mios * (perturbed-_pios) / di.GetPrimaryScale();
      scan.SetDerivParam(di.GetPrimaryIndex(),p,whichp,true);
    }
    else {
      double p = scan.GetDerivParam(di.GetPrimaryIndex(),whichp);
      scan.SetDerivParam(di.GetPrimaryIndex(),p+di.GetPrimaryScale(),whichp);
      EddyInternalGpuUtils::transform_model_to_scan_space(pred,scan,susc,true,perturbed,mask,jac,grad);
      _derivs[di.GetPrimaryIndex()] = _mios * (perturbed-_pios) / di.GetPrimaryScale();
      scan.SetDerivParam(di.GetPrimaryIndex(),p,whichp);
    }
    // Next calculate any secondary/modulated derivatives
    if (di.IsSliceMod()) {
      for (unsigned int j=0; j<di.NSecondary(); j++) {
	EDDY::SliceDerivModulator sdm = di.GetSliceModulator(j);
	get_slice_modulated_deriv(_derivs,mask,di.GetPrimaryIndex(),di.GetSecondaryIndex(j),sdm);
      }
    }
    else if (di.IsSpatiallyMod()) {
      for (unsigned int j=0; j<di.NSecondary(); j++) {
	EDDY::SpatialDerivModulator sdm = di.GetSpatialModulator(j);
	if (offset_deriv.Size() && di.GetPrimaryIndex() == scan.NDerivs()) {
	  get_spatially_modulated_deriv(_derivs,mask,di.GetPrimaryIndex(),di.GetSecondaryIndex(j),sdm,offset_deriv);
	}
	else {
	  get_spatially_modulated_deriv(_derivs,mask,di.GetPrimaryIndex(),di.GetSecondaryIndex(j),sdm,skrutt);
	}
      }
    }
  }

  return;
} EddyCatch

void DerivativeCalculator::get_lower_bound_indicies(// Input
						    const CudaVolume4D&         field,
						    const EDDY::AcqPara&        acqp,
						    const CudaVolume&           inmask,
						    // Output
						    thrust::device_vector<int>& lbindx,
						    CudaVolume&                 omask) const EddyTry
{
  if (lbindx.size() != field.Size()) {
    lbindx.resize(field.Size());
  }
  int tpb = field.Size(0);
  int nblocks = field.Size(2);
  if (acqp.PhaseEncodeVector()(1) != 0.0) { // If PE in x
    EddyKernels::get_lower_bound_of_inverse_x<<<nblocks,tpb>>>(field.GetPtr(0),inmask.GetPtr(),field.Size(0),field.Size(1),field.Size(2),
							       thrust::raw_pointer_cast(lbindx.data()),omask.GetPtr(),nblocks*tpb);
    EddyCudaHelperFunctions::CudaSync("EddyKernels::get_lower_bound_of_inverse_x");
  }
  else if (acqp.PhaseEncodeVector()(2) != 0.0) { // If PE in y
    EddyKernels::get_lower_bound_of_inverse_y<<<nblocks,tpb>>>(field.GetPtr(1),inmask.GetPtr(),field.Size(0),field.Size(1),field.Size(2),
							       thrust::raw_pointer_cast(lbindx.data()),omask.GetPtr(),nblocks*tpb);
    EddyCudaHelperFunctions::CudaSync("EddyKernels::get_lower_bound_of_inverse_y");
  }
  else throw EddyException("DerivativeCaclulator::get_lower_bound_indicies: Invalid phase encode vector");
  return;
} EddyCatch

void DerivativeCalculator::get_spatially_modulated_deriv(// Input/Output
                                                         CudaVolume4D&                derivs,
							 // Input
							 const CudaVolume&            mask,
							 unsigned int                 primi,
							 unsigned int                 scndi,
							 const SpatialDerivModulator& sdm,
							 const CudaVolume&            offset) const EddyTry
{
  std::vector<unsigned int> mod = sdm.GetModulation();
  int tpb = derivs.Size(0);
  int nblocks = derivs.Size(2);
  const float *inptr = nullptr;
  if (offset.Size()) inptr = offset.GetPtr();
  else inptr = derivs.GetPtr(primi);
  float *outptr = derivs.GetPtr(scndi);

  if (mod[0]) {
    for (unsigned int i=0; i<mod[0]; i++) {
      EddyKernels::x_modulate_deriv<<<nblocks,tpb>>>(inptr,mask.GetPtr(),derivs.Size(0),derivs.Size(1),
						     derivs.Size(2),derivs.Vxs(0),outptr,tpb*nblocks);
      EddyCudaHelperFunctions::CudaSync("DerivativeCalculator::get_spatially_modulated_deriv::x_modulate_deriv");
      inptr = outptr;
    }
  }
  if (mod[1]) {
    for (unsigned int i=0; i<mod[1]; i++) {
      EddyKernels::y_modulate_deriv<<<nblocks,tpb>>>(inptr,mask.GetPtr(),derivs.Size(0),derivs.Size(1),
						     derivs.Size(2),derivs.Vxs(1),outptr,tpb*nblocks);
      EddyCudaHelperFunctions::CudaSync("DerivativeCalculator::get_spatially_modulated_deriv::y_modulate_deriv");
      inptr = outptr;
    }
  }
  if (mod[2]) {
    for (unsigned int i=0; i<mod[2]; i++) {
      EddyKernels::z_modulate_deriv<<<nblocks,tpb>>>(inptr,mask.GetPtr(),derivs.Size(0),derivs.Size(1),
						     derivs.Size(2),derivs.Vxs(2),outptr,tpb*nblocks);
      EddyCudaHelperFunctions::CudaSync("DerivativeCalculator::get_spatially_modulated_deriv::z_modulate_deriv");
      inptr = outptr;
    }
  }
  return;
} EddyCatch

*/
