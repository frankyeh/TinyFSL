/////////////////////////////////////////////////////////////////////
///
/// \file EddyGpuUtils.cu
/// \brief Definitions of static class with collection of GPU routines used in the eddy project
///
/// \author Jesper Andersson & Moises Hernandez
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
#include <cuda.h>
#pragma push
#pragma diag_suppress = code_is_unreachable // Supress warnings from armawrap
#pragma diag_suppress = expr_has_no_effect  // Supress warnings from boost
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#pragma pop
#include "miscmaths/miscmaths.h"
#include "EddyCudaHelperFunctions.h"
#include "EddyInternalGpuUtils.h"
#include "EddyHelperClasses.h"
#include "DiffusionGP.h"
#include "b0Predictor.h"
#include "ECScanClasses.h"
#include "EddyUtils.h"
#include "EddyGpuUtils.h"
#include "EddyKernels.h"

using namespace EDDY;

std::shared_ptr<DWIPredictionMaker> EddyGpuUtils::LoadPredictionMaker(// Input
								      const EddyCommandLineOptions& clo,
								      ScanType                      st,
								      const ECScanManager&          sm,
								      unsigned int                  iter,
								      float                         fwhm,
								      // Output
								      NEWIMAGE::volume<float>&      mask,
								      // Optional input
								      bool                          use_orig) EddyTry
{
  std::shared_ptr<DWIPredictionMaker>  pmp;                                 // Prediction Maker Pointer
  if (st==DWI) { // If diffusion weighted data
    std::shared_ptr<KMatrix> K;
    if (clo.CovarianceFunction() == Spherical) K = std::shared_ptr<SphericalKMatrix>(new SphericalKMatrix(clo.DontCheckShelling()));
    else if (clo.CovarianceFunction() == Exponential) K = std::shared_ptr<ExponentialKMatrix>(new ExponentialKMatrix(clo.DontCheckShelling()));
    else if (clo.CovarianceFunction() == NewSpherical) K = std::shared_ptr<NewSphericalKMatrix>(new NewSphericalKMatrix(clo.DontCheckShelling()));
    else throw EddyException("LoadPredictionMaker: Unknown covariance function");
    std::shared_ptr<HyParCF> hpcf;
    std::shared_ptr<HyParEstimator> hpe;
    if (clo.HyperParFixed()) hpe = std::shared_ptr<FixedValueHyParEstimator>(new FixedValueHyParEstimator(clo.HyperParValues()));
    else {
      if (clo.HyParCostFunction() == CC) hpe = std::shared_ptr<CheapAndCheerfulHyParEstimator>(new CheapAndCheerfulHyParEstimator(clo.NVoxHp(),clo.InitRand()));
      else {
	if (clo.HyParCostFunction() == MML) hpcf = std::shared_ptr<MMLHyParCF>(new MMLHyParCF);
	else if (clo.HyParCostFunction() == CV) hpcf = std::shared_ptr<CVHyParCF>(new CVHyParCF);
	else if (clo.HyParCostFunction() == GPP) hpcf = std::shared_ptr<GPPHyParCF>(new GPPHyParCF);
	else throw EddyException("LoadPredictionMaker: Unknown hyperparameter cost-function");
	hpe = std::shared_ptr<FullMontyHyParEstimator>(new FullMontyHyParEstimator(hpcf,clo.HyParFudgeFactor(),clo.NVoxHp(),clo.InitRand(),clo.VeryVerbose()));
      }
    }
    pmp = std::shared_ptr<DWIPredictionMaker>(new DiffusionGP(K,hpe));  // GP
  }
  else pmp = std::shared_ptr<DWIPredictionMaker>(new b0Predictor);          // Silly mean predictor
  pmp->SetNoOfScans(sm.NScans(st));
  mask = sm.Scan(0,ANY).GetIma(); EddyUtils::SetTrilinearInterp(mask); mask = 1.0;

  EddyCudaHelperFunctions::InitGpu();
  EddyInternalGpuUtils::load_prediction_maker(clo,st,sm,iter,fwhm,use_orig,pmp,mask);

  if (clo.DebugLevel() > 2 && st==DWI) {
    char fname[256];
    sprintf(fname,"EDDY_DEBUG_K_Mat_Data_%02d",iter);
    pmp->WriteMetaData(fname);
  }

  return(pmp);
} EddyCatch

void EddyGpuUtils::MakeScatterBrainPredictions(// Input
					       const EddyCommandLineOptions& clo,
					       const ECScanManager&          sm,
					       const std::vector<double>&    hypar,
					       // Output
					       NEWIMAGE::volume4D<float>&    pred,
					       // Optional input
					       bool                        vwbvrot) EddyTry
{
  EddyInternalGpuUtils::make_scatter_brain_predictions(clo,sm,hypar,pred,vwbvrot);
} EddyCatch

/*
void EddyGpuUtils::UpdatePredictionMaker(// Input
					 const EddyCommandLineOptions&          clo,
					 ScanType                               st,
					 const ECScanManager&                   sm,
					 const ReplacementManager&              rm,
					 const NEWIMAGE::volume<float>&         mask,
					 // Input/Output
					 std::shared_ptr<DWIPredictionMaker>  pmp)
{
  EddyCudaHelperFunctions::InitGpu();
  EddyInternalGpuUtils::update_prediction_maker(clo,st,sm,rm,mask,pmp);

  return;
}
*/

NEWIMAGE::volume<float> EddyGpuUtils::GetUnwarpedScan(// Input
						      const EDDY::ECScan&                               scan,
						      std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
						      std::shared_ptr<const NEWIMAGE::volume<float> >   bias,
						      bool                                              use_orig,
						      // Optional output
						      NEWIMAGE::volume<float>                           *omask) EddyTry
{
  EDDY::CudaVolume cuda_susc;
  if (susc) cuda_susc = *susc;
  EDDY::CudaVolume cuda_bias;
  if (bias) cuda_bias = *bias;
  EDDY::CudaVolume empty;
  EDDY::CudaVolume uwscan(scan.GetIma(),false);
  if (omask) {
    EDDY::CudaVolume tmpmask(*omask,false);
    EddyInternalGpuUtils::get_unwarped_scan(scan,cuda_susc,cuda_bias,empty,true,use_orig,uwscan,tmpmask);
    *omask = tmpmask.GetVolume();
  }
  else {
    EDDY::CudaVolume tmpmask;
    EddyInternalGpuUtils::get_unwarped_scan(scan,cuda_susc,cuda_bias,empty,true,use_orig,uwscan,tmpmask);
  }
  return(uwscan.GetVolume());
} EddyCatch

NEWIMAGE::volume<float> EddyGpuUtils::GetUnwarpedScan(// Input
						      const EDDY::ECScan&                               scan,
						      std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
						      std::shared_ptr<const NEWIMAGE::volume<float> >   bias,
						      const NEWIMAGE::volume<float>&                    pred,
						      bool                                              use_orig,
						      // Optional output
						      NEWIMAGE::volume<float>                           *omask) EddyTry
{
  if (!scan.IsSliceToVol()) {
    std::cout << "EddyGpuUtils::GetUnwarpedScan: Warning, it does not make sense to supply pred for volumetric resampling" << std::endl;
  }
  if (scan.GetPolation().GetS2VInterp() != NEWIMAGE::spline) {
    throw EddyException("EddyGpuUtils::GetUnwarpedScan: use of prediction cannot be combined with trilinear interpolation");
  }
  EDDY::CudaVolume cuda_susc;
  if (susc) cuda_susc = *susc;
  EDDY::CudaVolume cuda_bias;
  if (bias) cuda_bias = *bias;
  EDDY::CudaVolume uwscan(scan.GetIma(),false);
  EDDY::CudaVolume cuda_pred = pred;
  if (omask) {
    EDDY::CudaVolume tmpmask(*omask,false);
    EddyInternalGpuUtils::get_unwarped_scan(scan,cuda_susc,cuda_bias,cuda_pred,true,use_orig,uwscan,tmpmask);
    *omask = tmpmask.GetVolume();
  }
  else {
    EDDY::CudaVolume tmpmask;
    EddyInternalGpuUtils::get_unwarped_scan(scan,cuda_susc,cuda_bias,cuda_pred,true,use_orig,uwscan,tmpmask);
  }
  return(uwscan.GetVolume());
} EddyCatch

NEWIMAGE::volume<float> EddyGpuUtils::GetVolumetricUnwarpedScan(// Input
								const EDDY::ECScan&                               scan,
								std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
								std::shared_ptr<const NEWIMAGE::volume<float> >   bias,
								bool                                              use_orig,
								// Optional output
								NEWIMAGE::volume<float>                           *omask,
								NEWIMAGE::volume4D<float>                         *deriv) EddyTry
{
  EDDY::CudaVolume cuda_susc;
  if (susc) cuda_susc = *susc;
  EDDY::CudaVolume cuda_bias;
  if (bias) cuda_bias = *bias;
  EDDY::CudaVolume empty;
  EDDY::CudaVolume uwscan(scan.GetIma(),false);
  if (omask && deriv) {
    EDDY::CudaVolume tmpmask(*omask,false);
    EDDY::CudaVolume4D tmpderiv(*deriv,false);
    EddyInternalGpuUtils::get_volumetric_unwarped_scan(scan,cuda_susc,cuda_bias,true,use_orig,uwscan,tmpmask,tmpderiv);
    *omask = tmpmask.GetVolume();
    *deriv = tmpderiv.GetVolume();
  }
  else if (omask) {
    EDDY::CudaVolume tmpmask(*omask,false);
    EDDY::CudaVolume4D tmpderiv;
    EddyInternalGpuUtils::get_volumetric_unwarped_scan(scan,cuda_susc,cuda_bias,true,use_orig,uwscan,tmpmask,tmpderiv);
    *omask = tmpmask.GetVolume();
  }
  else if (deriv) {
    EDDY::CudaVolume tmpmask;
    EDDY::CudaVolume4D tmpderiv(*deriv,false);
    EddyInternalGpuUtils::get_volumetric_unwarped_scan(scan,cuda_susc,cuda_bias,true,use_orig,uwscan,tmpmask,tmpderiv);
    *deriv = tmpderiv.GetVolume();
  }
  else {
    EDDY::CudaVolume tmpmask;
    EDDY::CudaVolume4D tmpderiv;
    EddyInternalGpuUtils::get_volumetric_unwarped_scan(scan,cuda_susc,cuda_bias,true,use_orig,uwscan,tmpmask,tmpderiv);
  }
  return(uwscan.GetVolume());
} EddyCatch

void EddyGpuUtils::GetMotionCorrectedScan(// Input
					  const EDDY::ECScan&       scan,
					  bool                      use_orig,
					  // Output
					  NEWIMAGE::volume<float>&  ovol,
					  // Optional output
					  NEWIMAGE::volume<float>   *omask) EddyTry
{
  EDDY::CudaVolume covol(scan.GetIma(),false);
  EDDY::CudaVolume comask;
  if (omask) {comask.SetHdr(covol); comask = 1.0; }
  EddyInternalGpuUtils::get_motion_corrected_scan(scan,use_orig,covol,comask);
  ovol = covol.GetVolume();
  if (omask) *omask = comask.GetVolume();
} EddyCatch

NEWIMAGE::volume<float> EddyGpuUtils::TransformModelToScanSpace(const EDDY::ECScan&                               scan,
								const NEWIMAGE::volume<float>&                    mima,
								std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
								bool                                              jacmod) EddyTry
{
  EDDY::CudaVolume mima_gpu = mima;
  EDDY::CudaVolume susc_gpu;
  if (susc != nullptr) susc_gpu = *susc;
  EDDY::CudaVolume mios(mima,false);
  EDDY::CudaVolume mask(mima,false); mask=1.0;
  EDDY::CudaVolume jac(mima,false);
  EDDY::CudaVolume4D skrutt4D;
  // cout << "Calling EddyInternalGpuUtils::transform_model_to_scan_space" << endl;
  EddyInternalGpuUtils::transform_model_to_scan_space(mima_gpu,scan,susc_gpu,jacmod,mios,mask,jac,skrutt4D);
  // cout << "Returning from EddyInternalGpuUtils::transform_model_to_scan_space" << endl;
  return(mios.GetVolume());
} EddyCatch

NEWIMAGE::volume4D<float> EddyGpuUtils::DerivativesForModelToScanSpaceTransform(const EDDY::ECScan&                               scan,
										const NEWIMAGE::volume<float>&                    mima,
										std::shared_ptr<const NEWIMAGE::volume<float> >   susc) EddyTry
{
  EDDY::CudaVolume mima_gpu = mima;
  EDDY::CudaVolume susc_gpu;
  if (susc != nullptr) susc_gpu = *susc;
  EDDY::CudaVolume4D derivs(mima,scan.NDerivs(),false);
  EddyInternalGpuUtils::get_partial_derivatives_in_scan_space(mima_gpu,scan,susc_gpu,EDDY::ALL,derivs);
  return(derivs.GetVolume());
} EddyCatch

NEWIMAGE::volume4D<float> EddyGpuUtils::DirectDerivativesForModelToScanSpaceTransform(const EDDY::ECScan&                               scan,
										      const NEWIMAGE::volume<float>&                    mima,
										      std::shared_ptr<const NEWIMAGE::volume<float> >   susc) EddyTry
{
  EDDY::CudaVolume mima_gpu = mima;
  EDDY::CudaVolume susc_gpu;
  if (susc != nullptr) susc_gpu = *susc;
  EDDY::CudaVolume4D derivs(mima,scan.NDerivs(),false);
  EddyInternalGpuUtils::get_direct_partial_derivatives_in_scan_space(mima_gpu,scan,susc_gpu,EDDY::ALL,derivs);
  return(derivs.GetVolume());
} EddyCatch

NEWIMAGE::volume<float> EddyGpuUtils::Smooth(const NEWIMAGE::volume<float>&  ima,
					     float                           fwhm) EddyTry
{
  EDDY::CudaVolume cuda_ima(ima,true);
  cuda_ima.Smooth(fwhm);
  return(cuda_ima.GetVolume());
} EddyCatch

DiffStatsVector EddyGpuUtils::DetectOutliers(// Input
					     const EddyCommandLineOptions&             clo,
					     ScanType                                  st,
					     const std::shared_ptr<DWIPredictionMaker> pmp,
					     const NEWIMAGE::volume<float>&            mask,
					     const ECScanManager&                      sm,
					     // Input/Output
					     ReplacementManager&                       rm) EddyTry
{
  EddyCudaHelperFunctions::InitGpu();
  DiffStatsVector  dsv(sm.NScans(st));
  EddyInternalGpuUtils::detect_outliers(clo,st,pmp,mask,sm,rm,dsv);
  return(dsv);
} EddyCatch

void EddyGpuUtils::ReplaceOutliers(// Input
				   const EddyCommandLineOptions&             clo,
				   ScanType                                  st,
				   const std::shared_ptr<DWIPredictionMaker> pmp,
				   const NEWIMAGE::volume<float>&            mask,
				   const ReplacementManager&                 rm,
				   bool                                      add_noise,
				   // Input/Output
				   ECScanManager&                            sm) EddyTry
{
  EddyCudaHelperFunctions::InitGpu();
  EddyInternalGpuUtils::replace_outliers(clo,st,pmp,mask,rm,add_noise,sm);
} EddyCatch

double EddyGpuUtils::MovAndECParamUpdate(// Input
					 const NEWIMAGE::volume<float>&                    pred,
					 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
					 std::shared_ptr<const NEWIMAGE::volume<float> >   bias,
					 const NEWIMAGE::volume<float>&                    pmask,
					 bool                                              cbs,
					 float                                             fwhm,
					 // Input/output
					 EDDY::ECScan&                                     scan) EddyTry
{
  EddyCudaHelperFunctions::InitGpu();
  return(EddyInternalGpuUtils::param_update(pred,susc,bias,pmask,EDDY::ALL,cbs,fwhm,0,0,0,scan,NULL));
} EddyCatch

double EddyGpuUtils::MovAndECParamUpdate(// Input
					 const NEWIMAGE::volume<float>&                    pred,
					 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
					 std::shared_ptr<const NEWIMAGE::volume<float> >   bias,
					 const NEWIMAGE::volume<float>&                    pmask,
					 bool                                              cbs,
					 float                                             fwhm,
					 // These inputs are for debug purposes only
					 unsigned int                                      scindex,
					 unsigned int                                      iter,
					 unsigned int                                      level,
					 // Input/output
					 EDDY::ECScan&                                     scan) EddyTry
{
  EddyCudaHelperFunctions::InitGpu();
  return(EddyInternalGpuUtils::param_update(pred,susc,bias,pmask,EDDY::ALL,cbs,fwhm,scindex,iter,level,scan,NULL));
} EddyCatch
