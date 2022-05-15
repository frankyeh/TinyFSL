/////////////////////////////////////////////////////////////////////
///
/// \file EddyInternalGpuUtils.cu
/// \brief Definitions of static class with collection of GPU routines used in the eddy project
///
/// \author Jesper Andersson
/// \version 1.0b, Nov., 2012.
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
// #include <cuda_profiler_api.h>
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
#include "DiffusionGP.h"
#include "b0Predictor.h"
#include "ECScanClasses.h"
#include "EddyUtils.h"
#include "EddyCudaHelperFunctions.h"
#include "CudaVolume.h"
#include "EddyKernels.h"
#include "EddyFunctors.h"
#include "EddyInternalGpuUtils.h"
#include "GpuPredictorChunk.h"
#include "StackResampler.h"
#include "DerivativeCalculator.h"
#include "CublasHandleManager.h"

using namespace EDDY;

void EddyInternalGpuUtils::load_prediction_maker(// Input
						 const EddyCommandLineOptions&        clo,
						 ScanType                             st,
						 const ECScanManager&                 sm,
						 unsigned int                         iter,
						 float                                fwhm,
						 bool                                 use_orig,
						 // Output
						 std::shared_ptr<DWIPredictionMaker>  pmp,
						 NEWIMAGE::volume<float>&             mask) EddyTry
{
  static Utilities::FSLProfiler prof("_"+std::string(__FILE__)+"_"+std::string(__func__));
  double total_key = prof.StartEntry("Total");
  if (sm.NScans(st)) {
    EDDY::CudaVolume omask(sm.Scan(0,st).GetIma(),false);
    omask.SetInterp(NEWIMAGE::trilinear); omask = 1.0;
    EDDY::CudaVolume tmpmask(omask,false);
    EDDY::CudaVolume uwscan;
    EDDY::CudaVolume empty;

    if (clo.Verbose()) std::cout << "Loading prediction maker";
    if (clo.VeryVerbose()) std::cout << std::endl << "Scan: ";
    double load_key = prof.StartEntry("Loading");
    for (int s=0; s<int(sm.NScans(st)); s++) {
      if (clo.VeryVerbose()) { std::cout << " " << s; std::cout.flush(); }
      EDDY::CudaVolume susc;
      if (sm.HasSuscHzOffResField()) susc = *(sm.GetSuscHzOffResField(s,st));
      EDDY::CudaVolume bias;
      if (sm.HasBiasField()) bias = *(sm.GetBiasField());
      EddyInternalGpuUtils::get_unwarped_scan(sm.Scan(s,st),susc,bias,empty,true,use_orig,uwscan,tmpmask);
      pmp->SetScan(uwscan.GetVolume(),sm.Scan(s,st).GetDiffPara(clo.RotateBVecsDuringEstimation()),s);
      omask *= tmpmask;
    }
    mask = omask.GetVolume();
    prof.EndEntry(load_key);

    if (clo.Verbose()) std::cout << std::endl << "Evaluating prediction maker model" << std::endl;
    double eval_key = prof.StartEntry("Evaluating");
    pmp->EvaluateModel(sm.Mask()*mask,fwhm,clo.VeryVerbose());
    prof.EndEntry(eval_key);
  }
  prof.EndEntry(total_key);

  return;
} EddyCatch

/*
void EddyInternalGpuUtils::update_prediction_maker(// Input
						   const EddyCommandLineOptions&          clo,
						   ScanType                               st,
						   const ECScanManager&                   sm,
						   const ReplacementManager&              rm,
						   const NEWIMAGE::volume<float>&         mask,
						   // Input/Output
						   std::shared_ptr<DWIPredictionMaker>  pmp)
{
  EDDY::CudaVolume susc;
  if (sm.GetSuscHzOffResField()) susc = *(sm.GetSuscHzOffResField());
  EDDY::CudaVolume uwscan;
  EDDY::CudaVolume skrutt;

  if (clo.Verbose()) std::cout << "Updating prediction maker";
  if (clo.VeryVerbose()) std::cout << std::endl << "Scan: ";
  for (unsigned int s=0; s<sm.NScans(st); s++) {
    if (rm.ScanHasOutliers(s)) {
      if (clo.VeryVerbose()) { std::cout << " " << s; std::cout.flush(); }
      EddyInternalGpuUtils::get_unwarped_scan(sm.Scan(s,st),susc,true,false,uwscan,skrutt);
      pmp->SetScan(uwscan.GetVolume(),sm.Scan(s,st).GetDiffPara(),s);
    }
  }

  if (clo.Verbose()) std::cout << std::endl << "Evaluating prediction maker model" << std::endl;
  pmp->EvaluateModel(sm.Mask()*mask,clo.FWHM(),clo.VeryVerbose());
}
*/

void EddyInternalGpuUtils::get_motion_corrected_scan(// Input
						     const EDDY::ECScan&     scan,
						     bool                    use_orig,
						     // Output
						     EDDY::CudaVolume&       oima,
						     // Optional output
						     EDDY::CudaVolume&       omask) EddyTry
{
  if (!oima.Size()) oima.SetHdr(scan.GetIma());
  else if (oima != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_motion_corrected_scan: scan<->oima mismatch");
  if (omask.Size() && omask != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_motion_corrected_scan: scan<->omask mismatch");
  EDDY::CudaVolume ima;
  EDDY::CudaVolume4D skrutt;
  if (use_orig) ima = scan.GetOriginalIma();
  else ima = scan.GetIma();
  if (scan.IsSliceToVol()) {
    std::vector<NEWMAT::Matrix> iR = EddyUtils::GetSliceWiseInverseMovementMatrices(scan);
    EddyInternalGpuUtils::affine_transform(ima,iR,oima,skrutt,omask);
  }
  else {
    // Transform image using inverse RB
    NEWMAT::Matrix iR = scan.InverseMovementMatrix();
    if (omask.Size()) omask = 1.0;
    EddyInternalGpuUtils::affine_transform(ima,iR,oima,skrutt,omask);
  }
  return;
} EddyCatch

void EddyInternalGpuUtils::get_unwarped_scan(// Input
					     const EDDY::ECScan&        scan,
					     const EDDY::CudaVolume&    susc,
					     const EDDY::CudaVolume&    bias,
					     const EDDY::CudaVolume&    pred,
					     bool                       jacmod,
					     bool                       use_orig,
					     // Output
					     EDDY::CudaVolume&          oima,
					     // Optional output
					     EDDY::CudaVolume&          omask) EddyTry
{
  if (!oima.Size()) oima.SetHdr(scan.GetIma());
  else if (oima != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_unwarped_scan: scan<->oima mismatch");
  if (susc.Size() && susc != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_unwarped_scan: scan<->susc mismatch");
  if (bias.Size() && bias != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_unwarped_scan: scan<->bias mismatch");
  if (omask.Size() && omask != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_unwarped_scan: scan<->omask mismatch");
  if (pred.Size() && !scan.IsSliceToVol()) throw EDDY::EddyException("EddyInternalGpuUtils::get_unwarped_scan: pred for volumetric does not make sense");
  EDDY::CudaVolume ima;
  if (use_orig) ima = scan.GetOriginalIma();
  else ima = scan.GetIma();
  if (bias.Size()) { // If we are to correct for receieve bias
    EDDY::CudaVolume4D idfield(ima,3,false);
    EDDY::CudaVolume mask1(ima,false); mask1 = 1.0;
    EDDY::CudaVolume empty;
    EddyInternalGpuUtils::field_for_model_to_scan_transform(scan,susc,idfield,empty,empty);
    EDDY::CudaVolume obias; // Biasfield sampled where each voxel in the scan "actually was"
    NEWMAT::IdentityMatrix I(4);
    EDDY::CudaVolume4D empty4D;
    EddyInternalGpuUtils::general_transform(bias,I,idfield,I,obias,empty4D,mask1);
    ima.DivideWithinMask(obias,mask1); // Correct ima for bias-field
  }
  EDDY::CudaVolume4D dfield(ima,3,false);
  EDDY::CudaVolume4D skrutt;
  EDDY::CudaVolume jac(ima,false);
  EDDY::CudaVolume mask2;
  if (omask.Size()) { mask2.SetHdr(jac); mask2 = 1.0; }
  EddyInternalGpuUtils::field_for_scan_to_model_transform(scan,susc,dfield,omask,jac);
  if (scan.IsSliceToVol()) {
    std::vector<NEWMAT::Matrix> iR = EddyUtils::GetSliceWiseInverseMovementMatrices(scan);
    EddyInternalGpuUtils::general_slice_to_vol_transform(ima,iR,dfield,jac,pred,jacmod,scan.GetPolation(),oima,mask2);
  }
  else {
    NEWMAT::Matrix iR = scan.InverseMovementMatrix();
    NEWMAT::IdentityMatrix I(4);
    EddyInternalGpuUtils::general_transform(ima,iR,dfield,I,oima,skrutt,mask2);
    if (jacmod) oima *= jac;
  }
  if (omask.Size()) {
    omask *= mask2;
    omask.SetInterp(NEWIMAGE::trilinear);
  }
} EddyCatch

void EddyInternalGpuUtils::get_volumetric_unwarped_scan(// Input
							const EDDY::ECScan&        scan,
							const EDDY::CudaVolume&    susc,
							const EDDY::CudaVolume&    bias,
							bool                       jacmod,
							bool                       use_orig,
							// Output
							EDDY::CudaVolume&          oima,
							// Optional output
							EDDY::CudaVolume&          omask,
							EDDY::CudaVolume4D&        deriv) EddyTry
{
  if (!oima.Size()) oima.SetHdr(scan.GetIma());
  else if (oima != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_volumetric_unwarped_scan: scan<->oima mismatch");
  if (susc.Size() && susc != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_volumetric_unwarped_scan: scan<->susc mismatch");
  if (bias.Size() && bias != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_volumetric_unwarped_scan: scan<->bias mismatch");
  if (omask.Size() && omask != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_volumetric_unwarped_scan: scan<->omask mismatch");
  if (deriv.Size() && deriv != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_volumetric_unwarped_scan: scan<->deriv mismatch");
  EDDY::CudaVolume ima;
  if (use_orig) ima = scan.GetOriginalIma();
  else ima = scan.GetIma();
  if (bias.Size()) { // If we are to correct for receieve bias
    EDDY::CudaVolume4D idfield(ima,3,false);
    EDDY::CudaVolume mask1(ima,false); mask1 = 1.0;
    EDDY::CudaVolume empty;
    EddyInternalGpuUtils::field_for_model_to_scan_transform(scan,susc,idfield,empty,empty);
    EDDY::CudaVolume obias; // Biasfield sampled where each voxel in the scan "actually was"
    NEWMAT::IdentityMatrix I(4);
    EDDY::CudaVolume4D empty4D;
    EddyInternalGpuUtils::general_transform(bias,I,idfield,I,obias,empty4D,mask1);
    ima.DivideWithinMask(obias,mask1); // Correct ima for bias-field
  }
  EDDY::CudaVolume4D dfield(ima,3,false);
  EDDY::CudaVolume jac(ima,false);
  EDDY::CudaVolume mask2;
  if (omask.Size()) { mask2.SetHdr(jac); mask2 = 1.0; }
  EddyInternalGpuUtils::field_for_scan_to_model_volumetric_transform(scan,susc,dfield,omask,jac);
  NEWMAT::Matrix iR = scan.InverseMovementMatrix();
  NEWMAT::IdentityMatrix I(4);
  EddyInternalGpuUtils::general_transform(ima,iR,dfield,I,oima,deriv,mask2);
  if (jacmod) oima *= jac;
  if (omask.Size()) {
    omask *= mask2;
    omask.SetInterp(NEWIMAGE::trilinear);
  }
} EddyCatch

void EddyInternalGpuUtils::detect_outliers(// Input
					   const EddyCommandLineOptions&             clo,
					   ScanType                                  st,
					   const std::shared_ptr<DWIPredictionMaker> pmp,
					   const NEWIMAGE::volume<float>&            pmask,
					   const ECScanManager&                      sm,
					   // Input/Output
					   ReplacementManager&                       rm,
					   DiffStatsVector&                          dsv) EddyTry
{
  static Utilities::FSLProfiler prof("_"+std::string(__FILE__)+"_"+std::string(__func__));
  double total_key = prof.StartEntry("Total");
  if (dsv.NScan() != sm.NScans(st)) throw EDDY::EddyException("EddyInternalGpuUtils::detect_outliers: dsv<->sm mismatch");
  if (clo.Verbose()) std::cout << "Checking for outliers" << std::endl;
  // Generate slice-wise stats on difference between observation and prediction
  for (GpuPredictorChunk c(sm.NScans(st),pmask); c<sm.NScans(st); c++) {
    std::vector<unsigned int> si = c.Indicies();
    EDDY::CudaVolume   pios(pmask,false);
    EDDY::CudaVolume   mios(pmask,false);
    EDDY::CudaVolume   mask(pmask,false);
    EDDY::CudaVolume   skrutt(pmask,false);
    EDDY::CudaVolume4D skrutt4D;
    if (clo.VeryVerbose()) std::cout << "Making predictions for scans: " << c << std::endl;
    std::vector<NEWIMAGE::volume<float> > cpred = pmp->Predict(si);
    if (clo.VeryVerbose()) { std::cout << "Checking scan: "; std::cout.flush(); }
    for (unsigned int i=0; i<si.size(); i++) {
      if (clo.VeryVerbose()) { std::cout << si[i] << " "; std::cout.flush(); }
      EDDY::CudaVolume gpred = cpred[i];
      // Transform prediction into observation space
      EDDY::CudaVolume susc;
      if (sm.HasSuscHzOffResField()) susc = *(sm.GetSuscHzOffResField(si[i],st));
      EddyInternalGpuUtils::transform_model_to_scan_space(gpred,sm.Scan(si[i],st),susc,true,pios,mask,skrutt,skrutt4D);
      // Transform binary mask into observation space
      CudaVolume bmask = sm.Mask();
      bmask *= pmask; bmask.SetInterp(NEWIMAGE::trilinear);
      EddyInternalGpuUtils::transform_model_to_scan_space(bmask,sm.Scan(si[i],st),susc,false,mios,mask,skrutt,skrutt4D);
      mios.Binarise(0.99); // Value above (arbitrary) 0.99 implies valid voxels
      mask *= mios;        // Volume and prediction mask falls within FOV
      // Calculate slice-wise stats from difference image
      DiffStats stats(sm.Scan(si[i],st).GetOriginalIma()-pios.GetVolume(),mask.GetVolume());
      dsv[si[i]] = stats;
    }
  }
  if (clo.VeryVerbose()) std::cout << std::endl;

  // Detect outliers and update replacement manager
  rm.Update(dsv);
  prof.EndEntry(total_key);

  return;
} EddyCatch

void EddyInternalGpuUtils::replace_outliers(// Input
					    const EddyCommandLineOptions&             clo,
					    ScanType                                  st,
					    const std::shared_ptr<DWIPredictionMaker> pmp,
					    const NEWIMAGE::volume<float>&            pmask,
					    const ReplacementManager&                 rm,
					    bool                                      add_noise,
					    // Input/Output
					    ECScanManager&                            sm) EddyTry
{
  static Utilities::FSLProfiler prof("_"+std::string(__FILE__)+"_"+std::string(__func__));
  double total_key = prof.StartEntry("Total");
  // Replace outlier slices with their predictions
  if (clo.VeryVerbose()) std::cout << "Replacing outliers with predictions" << std::endl;
  for (unsigned int s=0; s<sm.NScans(st); s++) {
    std::vector<unsigned int> ol = rm.OutliersInScan(s);
    if (ol.size()) { // If this scan has outlier slices
      if (clo.VeryVerbose()) std::cout << "Scan " << s << " has " << ol.size() << " outlier slices" << std::endl;
      EDDY::CudaVolume pred = pmp->Predict(s,true); // Make prediction
      EDDY::CudaVolume pios(pred,false);
      EDDY::CudaVolume mios(pred,false);
      EDDY::CudaVolume mask(pred,false);
      EDDY::CudaVolume jac(pred,false);
      EDDY::CudaVolume   skrutt;;
      EDDY::CudaVolume4D skrutt4D;
      EDDY::CudaVolume susc;
      if (sm.HasSuscHzOffResField()) susc = *(sm.GetSuscHzOffResField(s,st));
      // Transform prediction into observation space
      EddyInternalGpuUtils::transform_model_to_scan_space(pred,sm.Scan(s,st),susc,true,pios,mask,jac,skrutt4D);
      // Transform binary mask into observation space
      EDDY::CudaVolume pmask_cuda = pmask; pmask_cuda.SetInterp(NEWIMAGE::trilinear);
      EddyInternalGpuUtils::transform_model_to_scan_space(pmask_cuda,sm.Scan(s,st),susc,false,mios,mask,skrutt,skrutt4D);
      mios.Binarise(0.9); // Value above (arbitrary) 0.9 implies valid voxels
      mask *= mios;        // Volume and prediction mask falls within FOV
      if (add_noise) {
        double vp = pmp->PredictionVariance(s,true);
	double ve = pmp->ErrorVariance(s);
	double stdev = std::sqrt(vp+ve) - std::sqrt(vp);
	EDDY::CudaVolume nvol(pios,false);
	nvol.MakeNormRand(0.0,stdev);
	pios += nvol;
      }
      sm.Scan(s,st).SetAsOutliers(pios.GetVolume(),mask.GetVolume(),ol);
    }
  }
  prof.EndEntry(total_key);

  return;
} EddyCatch

void EddyInternalGpuUtils::field_for_scan_to_model_transform(// Input
							     const EDDY::ECScan&            scan,
							     const EDDY::CudaVolume&        susc,
							     // Output
							     EDDY::CudaVolume4D&            dfield,
							     // Optional output
							     EDDY::CudaVolume&              omask,
							     EDDY::CudaVolume&              jac) EddyTry
{
  if (susc.Size() && susc != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->susc mismatch");
  if (dfield.Size() && dfield != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->dfield mismatch");
  if (omask.Size() && omask != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->omask mismatch");
  if (jac.Size() && jac != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->jac mismatch");
  // Get EC field
  EDDY::CudaVolume  ec;
  EddyInternalGpuUtils::get_ec_field(scan,ec);
  // omask defines where the transformed EC map is valid
  if (omask.Size()) { omask = 1.0; omask.SetInterp(NEWIMAGE::trilinear); }
  EDDY::CudaVolume  tot(ec,false); tot = 0.0;
  if (scan.IsSliceToVol()) {
    // Original code
    /*
    std::vector<NEWMAT::Matrix> iR = EddyUtils::GetSliceWiseInverseMovementMatrices(scan);
    EDDY::CudaVolume4D skrutt;
    EddyInternalGpuUtils::affine_transform(ec,iR,tot,skrutt,omask);
    */
    // New code
    std::vector<NEWMAT::Matrix> iR = EddyUtils::GetSliceWiseInverseMovementMatrices(scan);
    NEWMAT::Matrix M1 = ec.Ima2WorldMatrix();
    std::vector<NEWMAT::Matrix> AA(iR.size());
    for (unsigned int i=0; i<iR.size(); i++) AA[i] = iR[i].i();
    NEWMAT::Matrix M2 = ec.World2ImaMatrix();
    EDDY::CudaImageCoordinates coord(ec.Size(0),ec.Size(1),ec.Size(2),false);
    EDDY::CudaVolume zcoordV(ec,false);
    EDDY::CudaVolume4D skrutt(ec,3,false);
    skrutt=0.0;
    coord.GetSliceToVolXYZCoord(M1,AA,skrutt,M2,zcoordV);
    // Then do a 2D resampling on regular grid
    ec.Sample(coord,tot);
    // Next re-sample the susc field in the z-direction and add to EC field
    if (susc.Size()) {
      EDDY::CudaImageCoordinates zcoord(ec.Size(0),ec.Size(1),ec.Size(2),false);
      zcoord.GetSliceToVolZCoord(M1,AA,skrutt,M2);
      EDDY::CudaVolume tmp(ec,false);
      susc.Sample(zcoord,tmp);
      tot += tmp;
    }
  }
  else {
    // Get RB matrix
    NEWMAT::Matrix    iR = scan.InverseMovementMatrix();
    EDDY::CudaVolume4D skrutt;
    // Transform EC field using RB
    EddyInternalGpuUtils::affine_transform(ec,iR,tot,skrutt,omask);
    // Add transformed EC and susc
    if (susc.Size()) tot += susc;
  }
  // Convert Hz-map to displacement field
  FieldGpuUtils::Hz2VoxelDisplacements(tot,scan.GetAcqPara(),dfield);
  // Get Jacobian of tot map
  if (jac.Size()) FieldGpuUtils::GetJacobian(dfield,scan.GetAcqPara(),jac);
  // Transform dfield from voxels to mm
  FieldGpuUtils::Voxel2MMDisplacements(dfield);
} EddyCatch

void EddyInternalGpuUtils::field_for_scan_to_model_volumetric_transform(// Input
									const EDDY::ECScan&            scan,
									const EDDY::CudaVolume&        susc,
									// Output
									EDDY::CudaVolume4D&            dfield,
									// Optional output
									EDDY::CudaVolume&              omask,
									EDDY::CudaVolume&              jac) EddyTry
{
  if (susc.Size() && susc != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->susc mismatch");
  if (dfield.Size() && dfield != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->dfield mismatch");
  if (omask.Size() && omask != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->omask mismatch");
  if (jac.Size() && jac != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->jac mismatch");
  // Get EC field
  EDDY::CudaVolume  ec;
  EddyInternalGpuUtils::get_ec_field(scan,ec);
  // omask defines where the transformed EC map is valid
  if (omask.Size()) { omask = 1.0; omask.SetInterp(NEWIMAGE::trilinear); }
  EDDY::CudaVolume  tot(ec,false); tot = 0.0;
  // Get RB matrix
  NEWMAT::Matrix    iR = scan.InverseMovementMatrix();
  EDDY::CudaVolume4D skrutt;
  // Transform EC field using RB
  EddyInternalGpuUtils::affine_transform(ec,iR,tot,skrutt,omask);
  // Add transformed EC and susc
  if (susc.Size()) tot += susc;
  // Convert Hz-map to displacement field
  FieldGpuUtils::Hz2VoxelDisplacements(tot,scan.GetAcqPara(),dfield);
  // Get Jacobian of tot map
  if (jac.Size()) FieldGpuUtils::GetJacobian(dfield,scan.GetAcqPara(),jac);
  // Transform dfield from voxels to mm
  FieldGpuUtils::Voxel2MMDisplacements(dfield);
} EddyCatch

double EddyInternalGpuUtils::param_update(// Input
					  const NEWIMAGE::volume<float>&                  pred,     // Prediction in model space
					  std::shared_ptr<const NEWIMAGE::volume<float> > susc,     // Susc-induced off-resonance field
					  std::shared_ptr<const NEWIMAGE::volume<float> > bias,     // Recieve bias field
					  const NEWIMAGE::volume<float>&                  pmask,    // Pre-defined mask in model space
					  EDDY::Parameters                                whichp,   // Which parameters to update
					  bool                                            cbs,      // Check (success of parameters) Before Set
					  float                                           fwhm,     // FWHM of smoothing
					  // These inputs are for debug purposes only
					  unsigned int                                    scindx,
					  unsigned int                                    iter,
					  unsigned int                                    level,
					  // Input/output
					  EDDY::ECScan&                                   scan,     // Scan we want to register to pred
					  // Optional output
					  NEWMAT::ColumnVector                            *rupdate) EddyTry // Vector of updates
{
  static Utilities::FSLProfiler prof("_"+std::string(__FILE__)+"_"+std::string(__func__));
  double total_key = prof.StartEntry("Total");
  // Put input images onto the GPU
  EDDY::CudaVolume pred_gpu(pred);
  // Transfer susceptibility field to GPU
  EDDY::CudaVolume susc_gpu;
  if (susc != nullptr) susc_gpu = *susc;
  // Transfer bias field to GPU
  EDDY::CudaVolume bias_gpu;
  if (bias != nullptr) bias_gpu = *bias;
  // Transfer binary input mask to GPU
  EDDY::CudaVolume pmask_gpu(pmask);
  pmask_gpu.SetInterp(NEWIMAGE::trilinear);
  // Define zero-size placeholders for use throughout function
  EDDY::CudaVolume   skrutt;
  EDDY::CudaVolume4D skrutt4D;

  double deriv_key = prof.StartEntry("Calculating derivatives");
  EDDY::DerivativeCalculator dc(pred_gpu,pmask_gpu,scan,susc_gpu,whichp,fwhm,DerivType::Mixed);
  prof.EndEntry(deriv_key);

  // Calculate XtX where X is a matrix whos columns are the partial derivatives
  NEWMAT::Matrix XtX;
  double XtX_key = prof.StartEntry("Calculating XtX");
  if (scan.IsSliceToVol()) XtX = EddyInternalGpuUtils::make_XtX_cuBLAS(dc.Derivatives());
  else XtX = EddyInternalGpuUtils::make_XtX(dc.Derivatives(),dc.MaskInScanSpace());
  prof.EndEntry(XtX_key);

  // Calculate difference image between observed and predicted
  EDDY::CudaVolume dima = dc.PredInScanSpace()-EDDY::CudaVolume(scan.GetIma());
  if (fwhm) dima.Smooth(fwhm,dc.MaskInScanSpace());
  // Calculate Xty where y is the difference between observed and predicted. X as above.
  NEWMAT::ColumnVector Xty = EddyInternalGpuUtils::make_Xty(dc.Derivatives(),dima,dc.MaskInScanSpace());
  // Get derivative and Hessian of regularisation (relevant only for slice-to-vol);
  NEWMAT::ColumnVector lHb = scan.GetRegGrad(whichp);
  NEWMAT::Matrix H = scan.GetRegHess(whichp);
  // Calculate mean sum of squares from difference image and add regularisation
  double masksum = dc.MaskInScanSpace().Sum();
  double mss = dima.SumOfSquares(dc.MaskInScanSpace()) / masksum + scan.GetReg(whichp);
  // Very mild Tikhonov regularisation to select solution with smaller norm
  double lambda = 1.0/masksum;
  NEWMAT::IdentityMatrix eye(XtX.Nrows());
  // Calculate update to parameters
  NEWMAT::ColumnVector update = -(XtX/masksum + H + lambda*eye).i()*(Xty/masksum + lHb);
  // Update parameters
  for (unsigned int i=0; i<scan.NDerivs(whichp); i++) {
    scan.SetDerivParam(i,scan.GetDerivParam(i,whichp)+update(i+1),whichp);
  }

  // Write warning if update doesn't make sense
  if (level && !EddyUtils::UpdateMakesSense(scan,update)) {
    std::cout << "EddyInternalGpuUtils::param_update: update doesn't make sense" << std::endl;
  }

  if (cbs) { // If we should check that new parameters actually decreased the cost-function
    double check_key = prof.StartEntry("Checking new parameters");
    EDDY::CudaVolume pios(pred,false);
    EDDY::CudaVolume jac(pred,false);
    EDDY::CudaVolume mask(pred,false); mask = 1.0;
    EddyInternalGpuUtils::transform_model_to_scan_space(pred_gpu,scan,susc_gpu,true,pios,mask,jac,skrutt4D);
    // Transform binary mask into observation space
    mask = 0.0;
    EDDY::CudaVolume mios(pmask,false);
    EddyInternalGpuUtils::transform_model_to_scan_space(pmask_gpu,scan,susc_gpu,false,mios,mask,skrutt,skrutt4D);
    mios.Binarise(0.99); // Value above (arbitrary) 0.99 implies valid voxels
    mask *= mios; // Volume and prediction mask falls within FOV
    EDDY::CudaVolume ndima = pios-EDDY::CudaVolume(scan.GetIma());
    if (fwhm) ndima.Smooth(fwhm,mask);
    double mss_au = ndima.SumOfSquares(mask) / mask.Sum() + scan.GetReg(whichp);
    if (std::isnan(mss_au) || mss_au > mss) { // If cost not decreased, set parameters back to what they were
      for (unsigned int i=0; i<scan.NDerivs(whichp); i++) {
	scan.SetDerivParam(i,scan.GetDerivParam(i,whichp)-update(i+1),whichp);
      }
      if (level) { // Write out info about failed update if it is a debug run
	std::cout << "EddyInternalGpuUtils::param_update: updates rejected" << std::endl;
	std::cout << "EddyInternalGpuUtils::param_update: original mss = " << mss << ", after update mss = " << mss_au << std::endl;
	std::cout.flush();
      }
    }
  }
  if (rupdate) *rupdate = update;
  prof.EndEntry(total_key);

  // Write debug information if requested.
  if (level) EddyInternalGpuUtils::write_debug_info_for_param_update(scan,scindx,iter,level,cbs,fwhm,dc,susc_gpu,
								     bias_gpu,pred_gpu,dima,pmask_gpu,XtX,Xty,update);

  return(mss);
} EddyCatch

void EddyInternalGpuUtils::write_debug_info_for_param_update(const EDDY::ECScan&               scan,
							     unsigned int                      scindx,
							     unsigned int                      iter,
							     unsigned int                      level,
							     bool                              cbs,
							     float                             fwhm,
							     const EDDY::DerivativeCalculator& dc,
							     const EDDY::CudaVolume&           susc,
							     const EDDY::CudaVolume&           bias,
							     const EDDY::CudaVolume&           pred,
							     const EDDY::CudaVolume&           dima,
							     const EDDY::CudaVolume&           pmask,
							     const NEWMAT::Matrix&             XtX,
							     const NEWMAT::ColumnVector&       Xty,
							     const NEWMAT::ColumnVector&       update) EddyTry
{
  // First do the cbs (check of success) if requested to
  // Next write debug info
  char fname[256], bname[256];
  EDDY::CudaVolume scratch;
  if (scan.IsSliceToVol()) strcpy(bname,"EDDY_DEBUG_S2V_GPU");
  else strcpy(bname,"EDDY_DEBUG_GPU");
  if (level>0) {
    sprintf(fname,"%s_masked_dima_%02d_%04d",bname,iter,scindx);
    scratch = dima * dc.MaskInScanSpace(); scratch.Write(fname);
  }
  if (level>1) {
    sprintf(fname,"%s_mask_%02d_%04d",bname,iter,scindx); dc.MaskInScanSpace().Write(fname);
    sprintf(fname,"%s_pios_%02d_%04d",bname,iter,scindx); dc.PredInScanSpace().Write(fname);
    sprintf(fname,"%s_pred_%02d_%04d",bname,iter,scindx); pred.Write(fname);
    sprintf(fname,"%s_dima_%02d_%04d",bname,iter,scindx); dima.Write(fname);
    sprintf(fname,"%s_jac_%02d_%04d",bname,iter,scindx); dc.JacInScanSpace().Write(fname);
    sprintf(fname,"%s_orig_%02d_%04d",bname,iter,scindx);
    scratch = scan.GetIma(); scratch.Write(fname);
    if (cbs) {
      EDDY::CudaVolume   new_pios;
      EDDY::CudaVolume   new_mios;
      EDDY::CudaVolume   new_mask;
      EDDY::CudaVolume   new_dima;
      EDDY::CudaVolume   new_jac(pred,false);
      EDDY::CudaVolume   scratch;
      EDDY::CudaVolume   skrutt;
      EDDY::CudaVolume4D skrutt4D;
      EddyInternalGpuUtils::transform_model_to_scan_space(pred,scan,susc,true,new_pios,new_mask,new_jac,skrutt4D);
      // Transform binary mask into observation space
      new_mask = 0.0;
      EddyInternalGpuUtils::transform_model_to_scan_space(pmask,scan,susc,false,new_mios,new_mask,skrutt,skrutt4D);
      new_mios.Binarise(0.99); // Value above (arbitrary) 0.99 implies valid voxels
      new_mask *= new_mios; // Volume and prediction mask falls within FOV
      new_dima = new_pios-EDDY::CudaVolume(scan.GetIma());
      if (fwhm) new_dima.Smooth(fwhm,new_mask);
      sprintf(fname,"%s_new_masked_dima_%02d_%04d",bname,iter,scindx);
      scratch = new_dima * new_mask; scratch.Write(fname);
      sprintf(fname,"%s_new_reverse_dima_%02d_%04d",bname,iter,scindx);
      EddyInternalGpuUtils::get_unwarped_scan(scan,susc,bias,skrutt,true,false,scratch,skrutt);
      scratch = pred - scratch; scratch.Write(fname);
      sprintf(fname,"%s_new_mask_%02d_%04d",bname,iter,scindx); new_mask.Write(fname);
      sprintf(fname,"%s_new_pios_%02d_%04d",bname,iter,scindx); new_pios.Write(fname);
      sprintf(fname,"%s_new_dima_%02d_%04d",bname,iter,scindx); new_dima.Write(fname);
      sprintf(fname,"%s_new_jac_%02d_%04d",bname,iter,scindx); new_jac.Write(fname);
    }
  }
  if (level>2) {
    sprintf(fname,"%s_mios_%02d_%04d",bname,iter,scindx); dc.MaskInScanSpace().Write(fname);
    sprintf(fname,"%s_pmask_%02d_%04d",bname,iter,scindx); pmask.Write(fname);
    sprintf(fname,"%s_derivs_%02d_%04d",bname,iter,scindx); dc.Derivatives().Write(fname);
    sprintf(fname,"%s_XtX_%02d_%04d.txt",bname,iter,scindx); MISCMATHS::write_ascii_matrix(fname,XtX,20);
    sprintf(fname,"%s_Xty_%02d_%04d.txt",bname,iter,scindx); MISCMATHS::write_ascii_matrix(fname,Xty,20);
    sprintf(fname,"%s_update_%02d_%04d.txt",bname,iter,scindx); MISCMATHS::write_ascii_matrix(fname,update,20);
  }
  return;
} EddyCatch

void EddyInternalGpuUtils::transform_model_to_scan_space(// Input
							 const EDDY::CudaVolume&       pred,
							 const EDDY::ECScan&           scan,
							 const EDDY::CudaVolume&       susc,
							 bool                          jacmod,
							 // Output
							 EDDY::CudaVolume&             oima,
							 EDDY::CudaVolume&             omask,
							 // Optional output
							 EDDY::CudaVolume&             jac,
							 EDDY::CudaVolume4D&           grad) EddyTry
{
  // static int cnt=1;
  // Some input checking
  if (oima != pred) oima.SetHdr(pred);
  if (omask != pred) omask.SetHdr(pred);
  if (jac.Size() && jac!=pred) throw EDDY::EddyException("EddyInternalGpuUtils::transform_model_to_scan_space: jac size mismatch");
  if (grad.Size() && grad!=pred) throw EDDY::EddyException("EddyInternalGpuUtils::transform_model_to_scan_space: grad size mismatch");
  if (jacmod && !jac.Size()) throw EDDY::EddyException("EddyInternalGpuUtils::transform_model_to_scan_space: jacmod can only be used with valid jac");
  EDDY::CudaVolume4D dfield(susc,3,false);
  EDDY::CudaVolume mask2(omask,false);
  NEWMAT::IdentityMatrix I(4);
  if (scan.IsSliceToVol()) {
    // Get field for the transform
    EddyInternalGpuUtils::field_for_model_to_scan_transform(scan,susc,dfield,omask,jac);
    // Get RB matrices, one per slice.
    std::vector<NEWMAT::Matrix> R = EddyUtils::GetSliceWiseForwardMovementMatrices(scan);
    std::vector<NEWMAT::Matrix> II(R.size()); for (unsigned int i=0; i<R.size(); i++) II[i] = I;
    // Transform prediction/model
    EddyInternalGpuUtils::general_transform(pred,II,dfield,R,oima,grad,mask2);
  }
  else {
    // Get field for the transform
    // auto start_get_field = std::chrono::system_clock::now();
    EddyInternalGpuUtils::field_for_model_to_scan_transform(scan,susc,dfield,omask,jac);
    // auto end_get_field = std::chrono::system_clock::now();
    // std::chrono::duration<double> duration = end_get_field-start_get_field;
    // std::cout << "EddyInternalGpuUtils::field_for_model_to_scan_transform(scan,susc,dfield,omask,jac); took " << duration.count() << " seconds" << std::endl;
    // Get RB matrix
    NEWMAT::Matrix R = scan.ForwardMovementMatrix();
    // Transform prediction/model
    EddyInternalGpuUtils::general_transform(pred,I,dfield,R,oima,grad,mask2);
    // char fname[256]; sprintf(fname,"field_%03d",cnt); dfield.Write(fname); cnt++;
  }
  omask *= mask2;
  omask.SetInterp(NEWIMAGE::trilinear);
  if (jacmod) oima *= jac;

  return;
} EddyCatch

void EddyInternalGpuUtils::field_for_model_to_scan_transform(// Input
							     const EDDY::ECScan&           scan,
							     const EDDY::CudaVolume&       susc,
							     // Output
							     EDDY::CudaVolume4D&           idfield,
							     EDDY::CudaVolume&             omask,
							     // Optional output
							     EDDY::CudaVolume&             jac) EddyTry
{
  // Some input checking
  if (idfield != scan.GetIma()) idfield.SetHdr(scan.GetIma(),3);
  if (omask.Size() && omask != scan.GetIma()) omask.SetHdr(scan.GetIma());
  if (susc.Size() && susc != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_model_to_scan_transform: susc size mismatch");
  if (jac.Size() && jac != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_model_to_scan_transform: jac size mismatch");

  EDDY::CudaVolume tot(scan.GetIma(),false);              // Total (EC and susc) field
  EDDY::CudaVolume mask(scan.GetIma(),false); mask = 1.0; // Defines where transformed susc field is valid
  EddyInternalGpuUtils::get_ec_field(scan,tot);
  if (susc.Size()) {
    EDDY::CudaVolume tsusc(susc,false);
    if (scan.IsSliceToVol()) {
      std::vector<NEWMAT::Matrix> R = EddyUtils::GetSliceWiseForwardMovementMatrices(scan);
      EDDY::CudaVolume4D skrutt;
      EddyInternalGpuUtils::affine_transform(susc,R,tsusc,skrutt,mask);
    }
    else {
      NEWMAT::Matrix R = scan.ForwardMovementMatrix();
      EDDY::CudaVolume4D skrutt;
      EddyInternalGpuUtils::affine_transform(susc,R,tsusc,skrutt,mask);
    }
    tot += tsusc;
  }
  // Convert Hz map to displacement field
  EDDY::CudaVolume4D dfield(tot,3,false);
  FieldGpuUtils::Hz2VoxelDisplacements(tot,scan.GetAcqPara(),dfield);
  // Invert displacement field
  FieldGpuUtils::InvertDisplacementField(dfield,scan.GetAcqPara(),mask,idfield,omask);
  // Get jacobian of inverted field
  if (jac.Size()) FieldGpuUtils::GetDiscreteJacobian(idfield,scan.GetAcqPara(),jac);
  // Transform field to mm displacements
  FieldGpuUtils::Voxel2MMDisplacements(idfield);
} EddyCatch

EDDY::CudaImageCoordinates EddyInternalGpuUtils::transform_coordinates_from_model_to_scan_space(// Input
												const EDDY::CudaVolume&     pred,
												const EDDY::ECScan&         scan,
												const EDDY::CudaVolume&     susc,
												// Output
												EDDY::CudaImageCoordinates& coord,
												// Optional Output
												EDDY::CudaVolume&           omask,
												EDDY::CudaVolume&           jac) EddyTry
{
  if (pred != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::transform_coordinates_from_model_to_scan_space: pred<->scan mismatch");
  if (susc.Size() && pred != susc) throw EDDY::EddyException("EddyInternalGpuUtils::transform_coordinates_from_model_to_scan_space: pred<->susc mismatch");
  if (omask.Size() && omask != pred) throw EDDY::EddyException("EddyInternalGpuUtils::transform_coordinates_from_model_to_scan_space: pred<->omask mismatch");
  if (jac.Size() && jac != pred) throw EDDY::EddyException("EddyInternalGpuUtils::transform_coordinates_from_model_to_scan_space: pred<->jac mismatch");
  // Get total field from scan
  EDDY::CudaVolume4D dfield;
  EddyInternalGpuUtils::field_for_model_to_scan_transform(scan,susc,dfield,omask,jac);
  // Get RB matrix
  NEWMAT::Matrix R = scan.ForwardMovementMatrix();
  // Convert matrices to mimic behaviour of warpfns:general_transform
  NEWMAT::Matrix A = pred.Ima2WorldMatrix();
  NEWMAT::Matrix M = pred.World2ImaMatrix() * R.i();
  // Transform coordinates using RB and inverted Tot map
  if (omask.Size()) {
    EDDY::CudaVolume mask2(omask,false);
    coord.Transform(A,dfield,M);
    pred.ValidMask(coord,mask2);
    omask *= mask2;
    omask.SetInterp(NEWIMAGE::trilinear);
  }
  else coord.Transform(A,dfield,M);

  return(coord);
} EddyCatch

void EddyInternalGpuUtils::get_partial_derivatives_in_scan_space(const EDDY::CudaVolume& pred,
								 const EDDY::ECScan&     scan,
								 const EDDY::CudaVolume& susc,
								 EDDY::Parameters        whichp,
								 EDDY::CudaVolume4D&     derivs) EddyTry
{
  EDDY::CudaVolume base(pred,false);
  EDDY::CudaVolume mask(pred,false);
  EDDY::CudaVolume basejac(pred,false);
  EDDY::CudaVolume4D grad(pred,3,false);
  EddyInternalGpuUtils::transform_model_to_scan_space(pred,scan,susc,true,base,mask,basejac,grad);
  EDDY::CudaImageCoordinates basecoord(pred.Size(0),pred.Size(1),pred.Size(2));
  EDDY::CudaVolume skrutt;
  EddyInternalGpuUtils::transform_coordinates_from_model_to_scan_space(pred,scan,susc,basecoord,skrutt,skrutt);
  if (derivs != pred || derivs.Size(3) != scan.NDerivs(whichp)) derivs.SetHdr(pred,scan.NDerivs(whichp));
  EDDY::CudaVolume jac(pred,false);
  EDDY::ECScan sc = scan;
  for (unsigned int i=0; i<sc.NDerivs(whichp); i++) {
    EDDY::CudaImageCoordinates diffcoord(pred.Size(0),pred.Size(1),pred.Size(2));
    double p = sc.GetDerivParam(i,whichp);
    sc.SetDerivParam(i,p+sc.GetDerivScale(i,whichp),whichp);
    EddyInternalGpuUtils::transform_coordinates_from_model_to_scan_space(pred,sc,susc,diffcoord,skrutt,jac);
    diffcoord -= basecoord;
    EddyInternalGpuUtils::make_deriv_from_components(diffcoord,grad,base,jac,basejac,sc.GetDerivScale(i,whichp),derivs,i);
    sc.SetDerivParam(i,p,whichp);
  }
  return;
} EddyCatch

void EddyInternalGpuUtils::get_direct_partial_derivatives_in_scan_space(const EDDY::CudaVolume& pred,
									const EDDY::ECScan&     scan,
									const EDDY::CudaVolume& susc,
									EDDY::Parameters        whichp,
									EDDY::CudaVolume4D&     derivs) EddyTry
{
  EDDY::CudaVolume base(pred,false);
  EDDY::CudaVolume mask(pred,false);
  EDDY::CudaVolume jac(pred,false);
  EDDY::CudaVolume4D grad;
  EddyInternalGpuUtils::transform_model_to_scan_space(pred,scan,susc,true,base,mask,jac,grad);
  if (derivs != pred || derivs.Size(3) != scan.NDerivs(whichp)) derivs.SetHdr(pred,scan.NDerivs(whichp));
  EDDY::CudaVolume perturbed(pred,false);
  EDDY::ECScan sc = scan;
  for (unsigned int i=0; i<sc.NDerivs(whichp); i++) {
    double p = sc.GetDerivParam(i,whichp);
    sc.SetDerivParam(i,p+sc.GetDerivScale(i,whichp),whichp);
    EddyInternalGpuUtils::transform_model_to_scan_space(pred,sc,susc,true,perturbed,mask,jac,grad);
    derivs[i] = (perturbed-base)/sc.GetDerivScale(i,whichp);
    sc.SetDerivParam(i,p,whichp);
  }
  return;
} EddyCatch

void EddyInternalGpuUtils::make_deriv_from_components(const EDDY::CudaImageCoordinates&  coord,
						      const EDDY::CudaVolume4D&          grad,
						      const EDDY::CudaVolume&            base,
						      const EDDY::CudaVolume&            jac,
						      const EDDY::CudaVolume&            basejac,
						      float                              dstep,
						      EDDY::CudaVolume4D&                deriv,
						      unsigned int                       indx) EddyTry
{
  int tpb = EddyInternalGpuUtils::threads_per_block_make_deriv;
  int nthreads = base.Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
  EddyKernels::make_deriv<<<nblocks,tpb>>>(base.Size(0),base.Size(1),base.Size(2),coord.XPtr(),coord.YPtr(),coord.ZPtr(),
					   grad.GetPtr(0),grad.GetPtr(1),grad.GetPtr(2),base.GetPtr(),jac.GetPtr(),
					   basejac.GetPtr(),dstep,deriv.GetPtr(indx),nthreads);
  EddyCudaHelperFunctions::CudaSync("EddyKernels::make_deriv");
  /*
  EddyKernels::make_deriv_first_part<<<nblocks,tpb>>>(base.Size(0),base.Size(1),base.Size(2),coord.XPtr(),coord.YPtr(),coord.ZPtr(),
						      grad.GetPtr(0),grad.GetPtr(1),grad.GetPtr(2),base.GetPtr(),jac.GetPtr(),
						      basejac.GetPtr(),dstep,deriv.GetPtr(indx),nthreads);
  EddyCudaHelperFunctions::CudaSync("EddyKernels::make_deriv_first_part");
  if (indx == 6) deriv.Write(indx,"derivative_first_part");

  EddyKernels::make_deriv_second_part<<<nblocks,tpb>>>(base.Size(0),base.Size(1),base.Size(2),coord.XPtr(),coord.YPtr(),coord.ZPtr(),
						      grad.GetPtr(0),grad.GetPtr(1),grad.GetPtr(2),base.GetPtr(),jac.GetPtr(),
						      basejac.GetPtr(),dstep,deriv.GetPtr(indx),nthreads);
  EddyCudaHelperFunctions::CudaSync("EddyKernels::make_deriv_second_part");
  if (indx == 6) { deriv.Write(indx,"derivative_second_part"); exit(0); }
  */
  return;
} EddyCatch

NEWMAT::Matrix EddyInternalGpuUtils::make_XtX(const EDDY::CudaVolume4D&  X,
					      const EDDY::CudaVolume&    mask) EddyTry
{
  thrust::device_vector<float> dXtX(X.Size(3)*X.Size(3));
  thrust::device_vector<float> masked(X.Size());
  for (int i=0; i<X.Size(3); i++) {
    try {
      thrust::copy(mask.Begin(),mask.End(),masked.begin());
      thrust::transform(X.Begin(i),X.End(i),masked.begin(),masked.begin(),thrust::multiplies<float>());
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in EddyInternalGpuUtils::make_XtX after copy and transform with index: " << i << ", and message: " << e.what() << std::endl;
      throw;
    }
    for (int j=i; j<X.Size(3); j++) {
      try {
	dXtX[j*X.Size(3)+i] = thrust::inner_product(masked.begin(),masked.end(),X.Begin(j),0.0f);
      }
      catch(thrust::system_error &e) {
	std::cerr << "thrust::system_error thrown in EddyInternalGpuUtils::make_XtX after inner_product with i = " << i << ", j = " << j << ", and message: " << e.what() << std::endl;
	throw;
      }
    }
  }

  thrust::host_vector<float> hXtX = dXtX;  // Whole matrix in one transfer
  NEWMAT::Matrix XtX(X.Size(3),X.Size(3));
  for (int i=0; i<X.Size(3); i++) {
    for (int j=i; j<X.Size(3); j++) {
      XtX(j+1,i+1) = hXtX[j*X.Size(3)+i];
      if (j!=i) XtX(i+1,j+1) = XtX(j+1,i+1);
    }
  }

  return(XtX);
} EddyCatch

NEWMAT::Matrix EddyInternalGpuUtils::make_XtX_cuBLAS(const EDDY::CudaVolume4D&  X) EddyTry
{
  float *dXtX;
  cudaError_t cudastat = cudaMalloc((void **)&dXtX,X.Size(3)*X.Size(3)*sizeof(float));
  if (cudastat != cudaSuccess) throw EddyException("EddyInternalGpuUtils::make_XtX_cuBLAS: Unable to allocate device memory: cudaMalloc returned an error: " + EddyCudaHelperFunctions::cudaError2String(cudastat));
  float one = 1.0; float zero = 0.0;
  cublasStatus_t cublastat = cublasSsyrk(CublasHandleManager::GetHandle(),CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,
					 X.Size(3),X.Size(),&one,X.GetPtr(),X.Size(),&zero,dXtX,X.Size(3));
  if (cublastat != CUBLAS_STATUS_SUCCESS) throw EddyException("EddyInternalGpuUtils::make_XtX_cuBLAS: cublasSsyrk error: " + EddyCudaHelperFunctions::cublasError2String(cublastat));
  float *hXtX = new float[X.Size(3)*X.Size(3)];
  cudastat = cudaMemcpy(hXtX,dXtX,X.Size(3)*X.Size(3)*sizeof(float),cudaMemcpyDeviceToHost);
  if (cudastat != cudaSuccess) throw EddyException("EddyInternalGpuUtils::make_XtX_cuBLAS: Unable to copy from device memory: cudaMemcpy returned an error: " + EddyCudaHelperFunctions::cudaError2String(cudastat));
  cudastat = cudaFree(dXtX);
  if (cudastat != cudaSuccess) throw EddyException("EddyInternalGpuUtils::make_XtX_cuBLAS: Unable to free device memory: cudaFree returned  an error: " + EddyCudaHelperFunctions::cudaError2String(cudastat));

  NEWMAT::Matrix rval(X.Size(3),X.Size(3));
  for (int c=0; c<X.Size(3); c++) {
    for (int r=c; r<X.Size(3); r++) {
      rval(r+1,c+1) = hXtX[c*X.Size(3)+r];
      if (r!=c) rval(c+1,r+1) = rval(r+1,c+1);
    }
  }
  delete[] hXtX;
  return(rval);
} EddyCatch

NEWMAT::ColumnVector EddyInternalGpuUtils::make_Xty(const EDDY::CudaVolume4D&  X,
						    const EDDY::CudaVolume&    y,
						    const EDDY::CudaVolume&    mask) EddyTry
{
  NEWMAT::ColumnVector Xty(X.Size(3));
  thrust::device_vector<float> masked(X.Size());
  try {
    thrust::copy(mask.Begin(),mask.End(),masked.begin());
    thrust::transform(y.Begin(),y.End(),masked.begin(),masked.begin(),thrust::multiplies<float>());
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in EddyInternalGpuUtils::make_Xty after copy and transform with message: " << e.what() << std::endl;
    throw;
  }
  for (int i=0; i<X.Size(3); i++) {
    try {
      Xty(i+1) = thrust::inner_product(masked.begin(),masked.end(),X.Begin(i),0.0f);
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in EddyInternalGpuUtils::make_Xty after inner_product with i = " << i << ", and with message: " << e.what() << std::endl;
      throw;
    }
  }
  return(Xty);
} EddyCatch

void EddyInternalGpuUtils::make_scatter_brain_predictions(// Input
							  const EddyCommandLineOptions& clo,
							  const ECScanManager&          sm,
							  const std::vector<double>&    hypar,
							  // Output
							  NEWIMAGE::volume4D<float>&    pred,
							  // Optional input
							  bool                          vwbvrot) EddyTry
{
  ScanType st = EDDY::DWI;
  const NEWIMAGE::volume<float>& timp = sm.Scan(0,st).GetIma(); // Scratch image ref

  // Get a grid of rotated bvecs for each slice and volume
  std::vector<std::vector<NEWMAT::ColumnVector> > bvecs(sm.NScans(st));
  for (unsigned int s=0; s<sm.NScans(st); s++) {
    bvecs[s].resize(timp.zsize());
    for (unsigned int sl=0; sl<bvecs[s].size(); sl++) {
      if (vwbvrot) bvecs[s][sl] = sm.Scan(s,st).GetDiffPara(true).bVec();
      else bvecs[s][sl] = sm.Scan(s,st).GetDiffPara(sl,true).bVec();
    }
  }
  // Get b-values for the different shells and indicies into those shells
  std::vector<DiffPara> dpv = sm.GetDiffParas(st);
  std::vector<double> grpb;                      // Vector of b-values for the different shells
  std::vector<unsigned int> grpi;                // Vector of indicies (one per dwi) into grpb
  std::vector<std::vector<unsigned int> > grps;  // grpb.size() number of vectors of vectors of indicies into dpv
  EddyUtils::GetGroups(dpv,grps,grpi,grpb);

  // Perform a "half-way" slice-to-vol reorientation for all scans.
  NEWIMAGE::volume4D<float> stacks(timp.xsize(),timp.ysize(),timp.zsize(),sm.NScans(st));
  NEWIMAGE::volume4D<float> masks(timp.xsize(),timp.ysize(),timp.zsize(),sm.NScans(st));
  NEWIMAGE::volume4D<float> zcoords(timp.xsize(),timp.ysize(),timp.zsize(),sm.NScans(st));
  NEWIMAGE::volume4D<float> shellmeans(timp.xsize(),timp.ysize(),timp.zsize(),grpb.size());
  NEWIMAGE::volume4D<float> shellcount(timp.xsize(),timp.ysize(),timp.zsize(),grpb.size());
  shellmeans = 0.0; shellcount = 0.0;
  NEWIMAGE::copybasicproperties(timp,stacks);
  NEWIMAGE::copybasicproperties(timp,masks);
  NEWIMAGE::copybasicproperties(timp,zcoords);
  EDDY::CudaVolume ima = sm.Scan(0,st).GetIma();
  EDDY::CudaVolume4D dfield(ima,3,false);
  EDDY::CudaVolume jac(ima,false);
  EDDY::CudaVolume susc;
  EDDY::CudaVolume hwima(ima,false);     // Stack of slices resampled in x and y only
  EDDY::CudaVolume zc(ima,false);        // z-ccordinates for the stack in hwima
  EDDY::CudaVolume oima(ima,false);      // "Finished" images resampled in all directions
  EDDY::CudaVolume fieldmask(ima,false); // Mask showing where field is valid
  EDDY::CudaVolume imamask(ima,false);   // Mask showing where image is valid
  for (unsigned int s=0; s<sm.NScans(st); s++) {
    ima = sm.Scan(s,st).GetIma();
    if (sm.HasSuscHzOffResField()) susc = *(sm.GetSuscHzOffResField(s,st));
    EddyInternalGpuUtils::field_for_scan_to_model_transform(sm.Scan(s,st),susc,dfield,fieldmask,jac);
    std::vector<NEWMAT::Matrix> iR = EddyUtils::GetSliceWiseInverseMovementMatrices(sm.Scan(s,st));
    EddyInternalGpuUtils::half_way_slice_to_vol_transform(ima,iR,dfield,jac,hwima,zc,oima,imamask);
    // EddyInternalGpuUtils::half_way_slice_to_vol_transform(ima,iR,dfield,jac,hwima,zc,imamask);
    stacks[s] = hwima.GetVolume();
    masks[s] = (imamask * fieldmask).GetVolume();
    zcoords[s] = zc.GetVolume();
    shellmeans[grpi[s]] += oima.GetVolume();
    shellcount[grpi[s]] += masks[s];
  }
  // Finish the shell mean images
  for (int k=0; k<shellmeans.zsize(); k++) {
    for (int j=0; j<shellmeans.ysize(); j++) {
      for (int i=0; i<shellmeans.xsize(); i++) {
	for (int s=0; s<shellmeans.tsize(); s++) {
	  if (shellcount(i,j,k,s) > 0) shellmeans(i,j,k,s) /= shellcount(i,j,k,s);
	  else shellmeans(i,j,k,s) = 0.0;
	}
      }
    }
  }

  // Mean correct all the values in stacks based on a linear interpolation in z
  for (int k=0; k<stacks.zsize(); k++) {
    for (int j=0; j<stacks.ysize(); j++) {
      for (int i=0; i<stacks.xsize(); i++) {
	for (int s=0; s<stacks.tsize(); s++) {
	  if (zcoords(i,j,k,s) > 0.0 && zcoords(i,j,k,s) < (stacks.zsize()-1) && masks(i,j,k,s) != 0.0) {
	    float m1 = shellmeans(i,j,std::floor(zcoords(i,j,k,s)),grpi[s]);
	    float m2 = shellmeans(i,j,std::ceil(zcoords(i,j,k,s)),grpi[s]);
	    if (m1 != 0.0 &&  m2 != 0.0) {
	      stacks(i,j,k,s) -= m1 * (1.0+std::floor(zcoords(i,j,k,s))-zcoords(i,j,k,s)) + m2 * (1.0+zcoords(i,j,k,s)-std::ceil(zcoords(i,j,k,s)));
	    }
	    else { stacks(i,j,k,s) = 0.0; masks(i,j,k,s) = 0.0; }
	  }
	  else { stacks(i,j,k,s) = 0.0; masks(i,j,k,s) = 0.0; }
	}
      }
    }
  }
  // Loop over all z-columns making predictions for all scans
  Stacks2YVecsAndWgts s2y(stacks.zsize(),stacks.tsize());
  if (clo.VeryVerbose()) { std::cout << "EddyInternalGpuUtils::make_scatter_brain_predictions: "; std::cout.flush(); }
  for (unsigned int j=0; j<stacks.ysize(); j++) {
    if (clo.VeryVerbose()) { std::cout << "* "; std::cout.flush(); }
    for (unsigned int i=0; i<stacks.xsize(); i++) {
      s2y.MakeVectors(stacks,masks,zcoords,i,j);
      for (unsigned int k=0; k<s2y.NVox(); k++) {
	Indicies2KMatrix i2k(bvecs,grpi,grpb,s2y.Indx(k),s2y.StdSqrtWgt(k),s2y.NVal(k),hypar);
	NEWMAT::ColumnVector predvec = i2k.GetKMatrix().i() * s2y.SqrtWgtYVec(k);
	for (unsigned int s=0; s<sm.NScans(st); s++) {
	  if (predvec.Nrows() > 20) {
	    pred(i,j,k,sm.GetDwi2GlobalIndexMapping(s)) = (i2k.GetkVector(sm.Scan(s,st).GetDiffPara(true).bVec(),grpi[s])*predvec).AsScalar();
	  }
	  else pred(i,j,k,sm.GetDwi2GlobalIndexMapping(s)) = 0.0;
	}
      }
    }
  }
  if (clo.VeryVerbose()) { std::cout << std::endl; std::cout.flush(); }
  // Add back shell means
  for (int k=0; k<stacks.zsize(); k++) {
    for (int j=0; j<stacks.ysize(); j++) {
      for (int i=0; i<stacks.xsize(); i++) {
	for (int s=0; s<stacks.tsize(); s++) {
	  pred(i,j,k,sm.GetDwi2GlobalIndexMapping(s)) += shellmeans(i,j,k,grpi[s]);
	}
      }
    }
  }
  return;
} EddyCatch


void FieldGpuUtils::Hz2VoxelDisplacements(const EDDY::CudaVolume&  hzfield,
					  const EDDY::AcqPara&     acqp,
					  EDDY::CudaVolume4D&      dfield) EddyTry
{
  if (dfield != hzfield || dfield.Size(3) != 3) dfield.SetHdr(hzfield,3);
  for (unsigned int i=0; i<3; i++) {
    if (acqp.PhaseEncodeVector()(i+1)) {
      thrust::transform(hzfield.Begin(),hzfield.End(),dfield.Begin(i),EDDY::MulByScalar<float>((acqp.PhaseEncodeVector())(i+1) * acqp.ReadOutTime()));
    }
    else thrust::fill(dfield.Begin(i),dfield.End(i),0.0);
  }
} EddyCatch

void FieldGpuUtils::Voxel2MMDisplacements(EDDY::CudaVolume4D&      dfield) EddyTry
{
  if (dfield.Size(3) != 3) throw EDDY::EddyException("FieldGpuUtils::Voxel2MMDisplacements: dfield.Size(3) must be 3");
  for (unsigned int i=0; i<dfield.Size(3); i++) {
    thrust::transform(dfield.Begin(i),dfield.End(i),dfield.Begin(i),EDDY::MulByScalar<float>(dfield.Vxs(i)));
  }
} EddyCatch

void FieldGpuUtils::InvertDisplacementField(// Input
					    const EDDY::CudaVolume4D&  dfield,
					    const EDDY::AcqPara&       acqp,
					    const EDDY::CudaVolume&    inmask,
					    // Output
					    EDDY::CudaVolume4D&        idfield,
					    EDDY::CudaVolume&          omask) EddyTry

{
  if (inmask != dfield) throw EDDY::EddyException("FieldGpuUtils::InvertDisplacementField: dfield<->inmask mismatch");
  if (acqp.PhaseEncodeVector()(1) && acqp.PhaseEncodeVector()(2)) throw EDDY::EddyException("FieldGpuUtils::InvertDisplacementField: Phase encode vector must have exactly one non-zero component");
  if (acqp.PhaseEncodeVector()(3)) throw EDDY::EddyException("FieldGpuUtils::InvertDisplacementField: Phase encode in z not allowed.");
  if (idfield != dfield) idfield.SetHdr(dfield); idfield = 0.0;
  if (omask != inmask) omask.SetHdr(inmask); omask = 0.0;
  int tpb = FieldGpuUtils::threads_per_block_invert_field;
  if (acqp.PhaseEncodeVector()(1)) {
    int nthreads = dfield.Size(1)*dfield.Size(2);
    int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
    EddyKernels::invert_displacement_field<<<nblocks,tpb>>>(dfield.GetPtr(0),inmask.GetPtr(),dfield.Size(0),dfield.Size(1),
							    dfield.Size(2),0,idfield.GetPtr(0),omask.GetPtr(),nthreads);
    EddyCudaHelperFunctions::CudaSync("EddyKernels::invert_displacement_field x");
  }
  else if (acqp.PhaseEncodeVector()(2)) {
    int nthreads = dfield.Size(0)*dfield.Size(2);
    int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
    EddyKernels::invert_displacement_field<<<nblocks,tpb>>>(dfield.GetPtr(1),inmask.GetPtr(),dfield.Size(0),dfield.Size(1),
							    dfield.Size(2),1,idfield.GetPtr(1),omask.GetPtr(),nthreads);
    EddyCudaHelperFunctions::CudaSync("EddyKernels::invert_displacement_field y");
  }
} EddyCatch

void FieldGpuUtils::GetJacobian(// Input
				const EDDY::CudaVolume4D&  dfield,
				const EDDY::AcqPara&       acqp,
				// Output
				EDDY::CudaVolume&          jac) EddyTry
{
  if (jac != dfield) jac.SetHdr(dfield);
  unsigned int cnt=0;
  for (unsigned int i=0; i<3; i++) if ((acqp.PhaseEncodeVector())(i+1)) cnt++;
  if (cnt != 1) throw EDDY::EddyException("FieldGpuUtils::GetJacobian: Phase encode vector must have exactly one non-zero component");
  unsigned int dir=0;
  for (dir=0; dir<3; dir++) if ((acqp.PhaseEncodeVector())(dir+1)) break;

  EDDY::CudaImageCoordinates coord(jac.Size(0),jac.Size(1),jac.Size(2),true);
  EDDY::CudaVolume tmpfield(jac,false);
  tmpfield.SetInterp(NEWIMAGE::spline);
  thrust::copy(dfield.Begin(dir),dfield.End(dir),tmpfield.Begin());
  EDDY::CudaVolume skrutt(jac,false);
  EDDY::CudaVolume4D grad(jac,3,false);
  tmpfield.Sample(coord,skrutt,grad);
  jac = 1.0;
  thrust::transform(jac.Begin(),jac.End(),grad.Begin(dir),jac.Begin(),thrust::plus<float>());

  return;
} EddyCatch

void FieldGpuUtils::GetDiscreteJacobian(// Input
					const EDDY::CudaVolume4D&  dfield,
					const EDDY::AcqPara&       acqp,
					// Output
					EDDY::CudaVolume&          jac) EddyTry
{
  if (jac != dfield) jac.SetHdr(dfield);
  unsigned int cnt=0;
  for (unsigned int i=0; i<3; i++) if ((acqp.PhaseEncodeVector())(i+1)) cnt++;
  if (cnt != 1) throw EDDY::EddyException("FieldGpuUtils::GetDiscreteJacobian: Phase encode vector must have exactly one non-zero component");

  unsigned int dir=0;
  for (dir=0; dir<3; dir++) if ((acqp.PhaseEncodeVector())(dir+1)) break;

  EDDY::CudaVolume skrutt;
  dfield.SampleTrilinearDerivOnVoxelCentres(dir,skrutt,jac);

  return;
} EddyCatch

void EddyInternalGpuUtils::general_transform(// Input
					     const EDDY::CudaVolume&    inima,
					     const NEWMAT::Matrix&      A,
					     const EDDY::CudaVolume4D&  dfield,
					     const NEWMAT::Matrix&      M,
					     // Output
					     EDDY::CudaVolume&          oima,
					     // Optional output
					     EDDY::CudaVolume4D&        deriv,
					     EDDY::CudaVolume&          omask) EddyTry
{
  if (oima != inima) oima.SetHdr(inima);
  if (dfield != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_transform: ima<->field mismatch");
  if (omask.Size() && omask != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_transform: wrong size omask");
  if (deriv.Size() && deriv != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_transform: wrong size deriv");
  // Convert matrices to mimic behaviour of warpfns:general_transform
  NEWMAT::Matrix AA = A.i() * inima.Ima2WorldMatrix();
  NEWMAT::Matrix MM = oima.World2ImaMatrix() * M.i();
  // Get transformed coordinates
  EDDY::CudaImageCoordinates coord(inima.Size(0),inima.Size(1),inima.Size(2),false);
  coord.Transform(AA,dfield,MM);
  // Interpolate image
  if (deriv.Size()) inima.Sample(coord,oima,deriv);
  else inima.Sample(coord,oima);
  // Calculate binary mask with 1 for valid voxels
  if (omask.Size()) inima.ValidMask(coord,omask);

  return;
} EddyCatch

void EddyInternalGpuUtils::general_transform(// Input
					     const EDDY::CudaVolume&             inima,
					     const std::vector<NEWMAT::Matrix>&  A,
					     const EDDY::CudaVolume4D&           dfield,
					     const std::vector<NEWMAT::Matrix>&  M,
					     // Output
					     EDDY::CudaVolume&                   oima,
					     // Optional output
					     EDDY::CudaVolume4D&                 deriv,
					     EDDY::CudaVolume&                   omask) EddyTry
{
  if (oima != inima) oima.SetHdr(inima);
  if (dfield != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_transform: ima<->field mismatch");
  if (omask.Size() && omask != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_transform: wrong size omask");
  if (deriv.Size() && deriv != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_transform: wrong size deriv");
  if (A.size() != inima.Size(2) || M.size() != inima.Size(2)) throw EDDY::EddyException("EddyInternalGpuUtils::general_transform: mismatched A or M vector");
  // Convert matrices to mimic behaviour of warpfns:general_transform
  std::vector<NEWMAT::Matrix> AA(A.size());
  for (unsigned int i=0; i<A.size(); i++) AA[i] = A[i].i() * inima.Ima2WorldMatrix();
  std::vector<NEWMAT::Matrix> MM(M.size());
  for (unsigned int i=0; i<M.size(); i++) MM[i] = oima.World2ImaMatrix() * M[i].i();
  // Get transformed coordinates
  EDDY::CudaImageCoordinates coord(inima.Size(0),inima.Size(1),inima.Size(2),false);
  coord.Transform(AA,dfield,MM);
  // Interpolate image
  if (deriv.Size()) inima.Sample(coord,oima,deriv);
  else inima.Sample(coord,oima);
  // Calculate binary mask with 1 for valid voxels
  if (omask.Size()) inima.ValidMask(coord,omask);

  return;
} EddyCatch

void EddyInternalGpuUtils::general_slice_to_vol_transform(// Input
							  const EDDY::CudaVolume&             inima,
							  const std::vector<NEWMAT::Matrix>&  A,
							  const EDDY::CudaVolume4D&           dfield,
							  const EDDY::CudaVolume&             jac,
							  const EDDY::CudaVolume&             pred,
							  bool                                jacmod,
							  const EDDY::PolationPara&           pp,
							  // Output
							  EDDY::CudaVolume&                   oima,
							  // Optional output
							  EDDY::CudaVolume&                   omask) EddyTry
{
  if (oima != inima) oima.SetHdr(inima);
  if (omask.Size() && omask != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_slice_to_vol_transform: wrong size omask");
  if (A.size() != inima.Size(2)) throw EDDY::EddyException("EddyInternalGpuUtils::general_slice_to_vol_transform: mismatched A vector");
  // Check if it might be an affine transform
  if (dfield.Size()) { // This means it isn't affine
    if (dfield != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_slice_to_vol_transform: ima<->field mismatch");
    if (jacmod && jac != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_slice_to_vol_transform: ima<->jac mismatch");
  }
  else { // So, affine transform intended
    if (jacmod) throw EDDY::EddyException("EddyInternalGpuUtils::general_slice_to_vol_transform: Invalid combination of jacmod and affine transform");
  }
  // Convert matrices to mimic behaviour of warpfns:general_transform
  NEWMAT::Matrix M1 = inima.Ima2WorldMatrix();
  std::vector<NEWMAT::Matrix> AA(A.size());
  for (unsigned int i=0; i<A.size(); i++) AA[i] = A[i].i();
  NEWMAT::Matrix M2 = oima.World2ImaMatrix();

  // This code removed as part of re-write 31/10-2016
  /*
  // First calculate a volume of z-coordinates to use for resampling of field
  EDDY::CudaImageCoordinates dcoord(inima.Size(0),inima.Size(1),inima.Size(2),false);
  dcoord.GetSliceToVolZCoord(M1,AA,dfield,M2);
  // Then resample the field
  EDDY::CudaVolume xfield(dfield.GetVolume(0));
  EDDY::CudaVolume yfield(dfield.GetVolume(1));
  EDDY::CudaVolume zfield(dfield.GetVolume(2));
  EDDY::CudaVolume4D rdfield(dfield,false);
  EDDY::CudaVolume tmp(inima,false);
  xfield.Sample(dcoord,tmp);
  rdfield.SetVolume(0,tmp);
  yfield.Sample(dcoord,tmp);
  rdfield.SetVolume(1,tmp);
  zfield.Sample(dcoord,tmp);
  rdfield.SetVolume(2,tmp);
  */

  // Then calculate x-, y- and z-coordinates for resampling of image
  EDDY::CudaImageCoordinates coord(inima.Size(0),inima.Size(1),inima.Size(2),false);
  EDDY::CudaVolume zcoordV(inima,false);
  // coord.GetSliceToVolXYZCoord(M1,AA,rdfield,M2,zcoordV);
  coord.GetSliceToVolXYZCoord(M1,AA,dfield,M2,zcoordV);
  // Then do a 2D resampling on regular grid in oima
  EDDY::CudaVolume resampled2D(inima,false);
  inima.Sample(coord,resampled2D);
  if (jacmod) resampled2D *= jac;
  EDDY::CudaVolume mask(inima,false);
  inima.ValidMask(coord,mask);
  // We now have a stack of slices (in resampled2D) resampled in-plane, and volume of z-coordinates for each "voxel" in resampled2D
  if (pred.Size()) {
    StackResampler sr(resampled2D,zcoordV,pred,mask,pp.GetSplineInterpLambda());
    oima = sr.GetResampledIma();
    if (omask.Size()) omask = sr.GetMask();
  }
  else {
    StackResampler sr(resampled2D,zcoordV,mask,pp.GetS2VInterp(),pp.GetSplineInterpLambda());
    oima = sr.GetResampledIma();
    if (omask.Size()) omask = sr.GetMask();
  }
  return;
} EddyCatch


void EddyInternalGpuUtils::half_way_slice_to_vol_transform(// Input
							  const EDDY::CudaVolume&             inima,
							  const std::vector<NEWMAT::Matrix>&  A,
							  const EDDY::CudaVolume4D&           dfield,
							  const EDDY::CudaVolume&             jac,
							  // Output
							  EDDY::CudaVolume&                   hwima,
							  EDDY::CudaVolume&                   zcoordV,
							  // Optional output
							  EDDY::CudaVolume&                   oima,
							  EDDY::CudaVolume&                   omask) EddyTry
{
  if (hwima != inima) hwima.SetHdr(inima);
  if (zcoordV != inima) zcoordV.SetHdr(inima);
  if (oima.Size() && oima != inima) throw EDDY::EddyException("EddyInternalGpuUtils::half_way_slice_to_vol_transform: wrong size oima");
  if (omask.Size() && omask != inima) throw EDDY::EddyException("EddyInternalGpuUtils::half_way_slice_to_vol_transform: wrong size omask");
  if (A.size() != inima.Size(2)) throw EDDY::EddyException("EddyInternalGpuUtils::half_way_slice_to_vol_transform: mismatched A vector");
  if (dfield != inima) throw EDDY::EddyException("EddyInternalGpuUtils::half_way_slice_to_vol_transform: ima<->field mismatch");
  if (jac != inima) throw EDDY::EddyException("EddyInternalGpuUtils::half_way_slice_to_vol_transform: ima<->jac mismatch");

  // Convert matrices to mimic behaviour of warpfns:general_transform
  NEWMAT::Matrix M1 = inima.Ima2WorldMatrix();
  std::vector<NEWMAT::Matrix> AA(A.size());
  for (unsigned int i=0; i<A.size(); i++) AA[i] = A[i].i();
  NEWMAT::Matrix M2 = hwima.World2ImaMatrix();

  // Then calculate x-, y- and z-coordinates for resampling of image
  EDDY::CudaImageCoordinates coord(inima.Size(0),inima.Size(1),inima.Size(2),false);
  coord.GetSliceToVolXYZCoord(M1,AA,dfield,M2,zcoordV);

  // Then do a 2D resampling on regular grid in oima
  inima.Sample(coord,hwima);
  hwima *= jac;
  if (omask.Size()) inima.ValidMask(coord,omask);

  // Optionally also output fully resampled volume
  if (oima.Size()) {
    StackResampler sr(hwima,zcoordV,omask,NEWIMAGE::trilinear,0.005);
    oima = sr.GetResampledIma();
  }

  // We now have a stack of slices (in resampled2D) resampled in-plane, and volume of z-coordinates for each "voxel" in zcoordV

  return;
} EddyCatch

void EddyInternalGpuUtils::affine_transform(const EDDY::CudaVolume&    inima,
					    const NEWMAT::Matrix&      R,
					    EDDY::CudaVolume&          oima,
					    EDDY::CudaVolume4D&        deriv,
					    EDDY::CudaVolume&          omask) EddyTry
{
  if (oima != inima) oima.SetHdr(inima);
  if (omask.Size() && omask != inima) throw EDDY::EddyException("EddyInternalGpuUtils::affine_transform: inima<->omask mismatch");
  if (deriv.Size() && deriv != inima) throw EDDY::EddyException("EddyInternalGpuUtils::affine_transform: inima<->deriv mismatch");
  // Convert matrix to mimic behaviour of warpfns:general_transform
  NEWMAT::Matrix A = oima.World2ImaMatrix() * R.i() * inima.Ima2WorldMatrix();
  // Get transformed coordinates
  EDDY::CudaImageCoordinates coord(inima.Size(0),inima.Size(1),inima.Size(2),false);
  coord.Transform(A);
  // Interpolate image
  if (deriv.Size()) inima.Sample(coord,oima,deriv);
  else inima.Sample(coord,oima);
  // Calculate binary mask with 1 for valid voxels
  if (omask.Size()) inima.ValidMask(coord,omask);

  return;
} EddyCatch

void EddyInternalGpuUtils::affine_transform(const EDDY::CudaVolume&             inima,
					    const std::vector<NEWMAT::Matrix>&  R,
					    EDDY::CudaVolume&                   oima,
					    EDDY::CudaVolume4D&                 deriv,
					    EDDY::CudaVolume&                   omask) EddyTry
{
  if (oima != inima) oima.SetHdr(inima);
  if (omask.Size() && omask != inima) throw EDDY::EddyException("EddyInternalGpuUtils::affine_transform: inima<->omask mismatch");
  if (deriv.Size() && deriv != inima) throw EDDY::EddyException("EddyInternalGpuUtils::affine_transform: inima<->deriv mismatch");
  if (R.size() != inima.Size(2)) throw EDDY::EddyException("EddyInternalGpuUtils::affine_transform: mismatched R vector");
  // Convert matrix to mimic behaviour of warpfns:general_transform
  std::vector<NEWMAT::Matrix> A(R.size());
  for (unsigned int i=0; i<R.size(); i++) A[i] = oima.World2ImaMatrix() * R[i].i() * inima.Ima2WorldMatrix();
  // Get transformed coordinates
  EDDY::CudaImageCoordinates coord(inima.Size(0),inima.Size(1),inima.Size(2),false);
  coord.Transform(A);
  // Interpolate image
  if (deriv.Size()) inima.Sample(coord,oima,deriv);
  else inima.Sample(coord,oima);
  // Calculate binary mask with 1 for valid voxels
  if (omask.Size()) inima.ValidMask(coord,omask);

  return;
} EddyCatch

void EddyInternalGpuUtils::get_ec_field(// Input
					const EDDY::ECScan&       scan,
					// Output
					EDDY::CudaVolume&         ecfield) EddyTry
{
  if (ecfield != scan.GetIma()) ecfield.SetHdr(scan.GetIma());
  // Transfer EC parameters onto device
  NEWMAT::ColumnVector epp = scan.GetParams(EDDY::EC);
  thrust::host_vector<float> epp_host(scan.NParam(EDDY::EC));
  for (unsigned int i=0; i<epp_host.size(); i++) epp_host[i] = epp(i+1);
  thrust::device_vector<float> epp_dev = epp_host; // EC parameters -> device

  int tpb = EddyInternalGpuUtils::threads_per_block_ec_field;
  int nthreads = ecfield.Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;

  if (scan.Model() == EDDY::NoEC) {
    ecfield = 0.0;
  }
  if (scan.Model() == EDDY::Linear) {
    EddyKernels::linear_ec_field<<<nblocks,tpb>>>(ecfield.GetPtr(),ecfield.Size(0),ecfield.Size(1),ecfield.Size(2),
						  ecfield.Vxs(0),ecfield.Vxs(1),ecfield.Vxs(2),
						  thrust::raw_pointer_cast(epp_dev.data()),epp_dev.size(),nthreads);
    EddyCudaHelperFunctions::CudaSync("EddyKernels::linear_ec_field");
  }
  else if (scan.Model() == EDDY::Quadratic) {
    EddyKernels::quadratic_ec_field<<<nblocks,tpb>>>(ecfield.GetPtr(),ecfield.Size(0),ecfield.Size(1),ecfield.Size(2),
						     ecfield.Vxs(0),ecfield.Vxs(1),ecfield.Vxs(2),
						     thrust::raw_pointer_cast(epp_dev.data()),epp_dev.size(),nthreads);
    EddyCudaHelperFunctions::CudaSync("EddyKernels::quadratic_ec_field");
  }
  else if (scan.Model() == EDDY::Cubic) {
    EddyKernels::cubic_ec_field<<<nblocks,tpb>>>(ecfield.GetPtr(),ecfield.Size(0),ecfield.Size(1),ecfield.Size(2),
						 ecfield.Vxs(0),ecfield.Vxs(1),ecfield.Vxs(2),
						 thrust::raw_pointer_cast(epp_dev.data()),epp_dev.size(),nthreads);
    EddyCudaHelperFunctions::CudaSync("EddyKernels::cubic_ec_field");
  }
  return;
} EddyCatch


// The rest of this file is dead code
/*
NEWMAT::Matrix EddyInternalGpuUtils::make_XtX_old(const EDDY::CudaVolume4D&  X,
						  const EDDY::CudaVolume&    mask) EddyTry
{
  NEWMAT::Matrix XtX(X.Size(3),X.Size(3));
  for (int i=0; i<X.Size(3); i++) {
    thrust::device_vector<float> masked(X.Size());
    try {
      thrust::copy(mask.Begin(),mask.End(),masked.begin());
      thrust::transform(X.Begin(i),X.End(i),masked.begin(),masked.begin(),thrust::multiplies<float>());
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in EddyInternalGpuUtils::make_XtX after copy and transform with index: " << i << ", and message: " << e.what() << std::endl;
      throw;
    }
    for (int j=i; j<X.Size(3); j++) {
      try {
	XtX(j+1,i+1) = thrust::inner_product(masked.begin(),masked.end(),X.Begin(j),0.0f);
      }
      catch(thrust::system_error &e) {
	std::cerr << "thrust::system_error thrown in EddyInternalGpuUtils::make_XtX after inner_product with i = " << i << ", j = " << j << ", and message: " << e.what() << std::endl;
	throw;
      }
      if (j!=i) XtX(i+1,j+1) = XtX(j+1,i+1);
    }
  }

  return(XtX);
} EddyCatch

*/

/*
NEWMAT::Matrix EddyInternalGpuUtils::make_XtX_new_2(const EDDY::CudaVolume4D&  X,
						    const EDDY::CudaVolume&    mask) EddyTry
{
  // unsigned int nunique = (X.Size(3)*(X.Size(3)+1)) / 2;

  thrust::device_vector<float> dXtX(X.Size(3)*X.Size(3));
  const float* *ima_ptr_vec;
  cudaMalloc((void ***) &ima_ptr_vec,X.Size(3) * sizeof(float *));
  const float* *ima_ptr_vec_host = new const float*[X.Size(3)];
  for (unsigned int i=0; i<X.Size(3); i++) ima_ptr_vec_host[i] = X.GetPtr(i);
  cudaError_t rval = cudaMemcpy(ima_ptr_vec,ima_ptr_vec_host,X.Size(3)*sizeof(float *),cudaMemcpyHostToDevice);

  int nblocks = static_cast<int>(X.Size(3));
  int tpb = static_cast<int>(X.Size(3));
  int nthreads = nblocks*tpb;
  EddyKernels::XtX<<<nblocks,tpb>>>(X.Size(3),X.Size(),ima_ptr_vec,mask.GetPtr(),thrust::raw_pointer_cast(dXtX.data()),nthreads);
  EddyCudaHelperFunctions::CudaSync("EddyKernels::XtX");
  thrust::host_vector<float> hXtX = dXtX;
  NEWMAT::Matrix XtX(X.Size(3),X.Size(3));
  for (int c=0; c<X.Size(3); c++) {
    for (int r=0; r<=c; r++) {
      XtX(r+1,c+1) = hXtX[c*X.Size(3)+r];
      if (r!=c) XtX(c+1,r+1) = XtX(r+1,c+1);
    }
  }

  rval = cudaFree(ima_ptr_vec);
  delete[] ima_ptr_vec_host;

  return(XtX);
} EddyCatch

*/
