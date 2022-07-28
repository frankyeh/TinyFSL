/*! \file eddy.cpp
    \brief Contains main() and some very high level functions for eddy
*/
#pragma GCC diagnostic ignored "-Wunknown-pragmas" // Ignore the OpenMP pragmas when not used
#if __cplusplus >= 201103L || __clang__
 #include <array>
using std::array;
#else
 #include <tr1/array>
using std::tr1::array;
#endif
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "utils/FSLProfiler.h"
#include "EddyHelperClasses.h"
#include "ECScanClasses.h"
#include "DiffusionGP.h"
#include "b0Predictor.h"
#include "EddyUtils.h"
#include "EddyCommandLineOptions.h"
#include "PostEddyCF.h"
#include "PostEddyAlignShellsFunctions.h"
#include "MoveBySuscCF.h"
#include "BiasFieldEstimator.h"
#include "eddy.h"
#ifdef COMPILE_GPU
#include "cuda/GpuPredictorChunk.h"
#include "cuda/EddyGpuUtils.h"
#endif

#include "TIPL/tipl.hpp"
using namespace std;
using namespace EDDY;

/// The entry point of eddy.
int main(int argc, char *argv[]) try
{
  // Parse comand line input
  EddyCommandLineOptions clo(argc,argv); // Command Line Options

  // Prime profiler if requested by user
  if (clo.LogTimings()) Utilities::FSLProfiler::SetProfilingOn(clo.LoggerFname());
  Utilities::FSLProfiler prof("_"+string(__FILE__)+"_"+string(__func__));
  double total_key = prof.StartEntry("Begins eddy run");

  // Read all available info
  if (clo.Verbose()) cout << "Reading images" << endl;
  ECScanManager sm(clo.ImaFname(),clo.MaskFname(),clo.AcqpFname(),clo.TopupFname(),clo.FieldFname(),clo.FieldMatFname(),
                   clo.BVecsFname(),clo.BValsFname(),clo.FirstLevelModel(),clo.b0_FirstLevelModel(),clo.Indicies(),
                   clo.PolationParameters(),clo.MultiBand(),clo.DontCheckShelling()); // Scan Manager
  if (clo.FillEmptyPlanes()) { if (clo.Verbose()) cout << "Filling empty planes" << endl; sm.FillEmptyPlanes(); }
  if (clo.ResamplingMethod() == LSR) {
    if (!sm.CanDoLSRResampling()) throw EddyException("These data do not support least-squares resampling");
  }
  if (clo.UseB0sToAlignShellsPostEddy() && !sm.B0sAreUsefulForPEAS()) {
    throw EddyException("These data do not support using b0s for Post Eddy Alignment of Shells");
  }
  if (clo.RefScanNumber()) sm.SetLocationReference(clo.RefScanNumber());

  // Write topup-field if debug flag is set
  if (clo.DebugLevel() && sm.HasSuscHzOffResField()) {
    std::string fname = "EDDY_DEBUG_susc_00_0000"; NEWIMAGE::write_volume(*(sm.GetSuscHzOffResField()),fname);
  }

  // Set initial parameters. This option is only for testing/debugging/personal use
  if (clo.InitFname() != std::string("")) {
    if (clo.RegisterDWI() && clo.Registerb0()) sm.SetParameters(clo.InitFname(),ANY);
    else if (clo.RegisterDWI()) sm.SetParameters(clo.InitFname(),DWI);
    else sm.SetParameters(clo.InitFname(),B0);
  }

  // Do the registration

  //////////////////////////////////////////////////////////////////////
  // The first, and possibly only, registration step is volume_to_volume
  //////////////////////////////////////////////////////////////////////

  double vol_key = prof.StartEntry("Calling DoVolumeToVolumeRegistration");
  if (clo.Verbose()) cout << "Performing volume-to-volume registration" << endl;
  ReplacementManager *dwi_rm=NULL;
  if (clo.EstimateMoveBySusc()) { // Restrict EC if we are to eventually estimate MBS
    EDDY::SecondLevelECModel b0_slm = clo.b0_SecondLevelModel();
    EDDY::SecondLevelECModel dwi_slm = clo.SecondLevelModel();
    if (clo.VeryVerbose()) cout << "Setting linear second level model" << endl;
    clo.Set_b0_SecondLevelModel(EDDY::Linear_2nd_lvl_mdl);
    clo.SetSecondLevelModel(EDDY::Linear_2nd_lvl_mdl);
    dwi_rm = DoVolumeToVolumeRegistration(clo,sm);
    if (clo.VeryVerbose()) cout << "Resetting second level model" << endl;
    clo.Set_b0_SecondLevelModel(b0_slm);
    clo.SetSecondLevelModel(dwi_slm);
  }
  else dwi_rm = DoVolumeToVolumeRegistration(clo,sm);
  prof.EndEntry(vol_key);

  // The remaining steps we only run if registration was actually run,
  // i.e. not if we read an initialisation file and did zero iterations.
  if (clo.NIter() > 0) {
    sm.ApplyLocationReference();

    // Write text-file with MI values if requested (testing/debugging)
    if (clo.PrintMIValues()) {
      if (clo.Verbose()) cout << "Writing MI values between shells" << endl;
      PrintMIValues(clo,sm,clo.MIPrintFname(),clo.PrintMIPlanes());
    }

    // Check for residual position differences between shells
    if (clo.RegisterDWI()) {
      double peas_key = prof.StartEntry("Calling PostEddyAlignShells");
      if (!clo.SeparateOffsetFromMovement() && !clo.AlignShellsPostEddy()) {
        if (clo.Verbose()) cout << "Checking shell alignment along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
        PEASUtils::PostEddyAlignShellsAlongPE(clo,false,sm);
        if (clo.Verbose()) cout << "Checking shell alignment (running PostEddyAlignShells)" << endl;
        PEASUtils::PostEddyAlignShells(clo,false,sm);
      }
      else if (clo.SeparateOffsetFromMovement() && !clo.AlignShellsPostEddy()) {
        if (clo.Verbose()) cout << "Aligning shells along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
        PEASUtils::PostEddyAlignShellsAlongPE(clo,true,sm);
        if (clo.Verbose()) cout << "Checking shell alignment (running PostEddyAlignShells)" << endl;
        PEASUtils::PostEddyAlignShells(clo,false,sm);
      }
      else if (clo.AlignShellsPostEddy()) {
        if (clo.Verbose()) cout << "Checking shell alignment along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
        PEASUtils::PostEddyAlignShellsAlongPE(clo,false,sm);
        if (clo.Verbose()) cout << "Aligning shells (running PostEddyAlignShells)" << endl;
        PEASUtils::PostEddyAlignShells(clo,true,sm);
      }
      prof.EndEntry(peas_key);
    }
  }

  //////////////////////////////////////////////////////////////////////
  // Next do a first round of estimating a bias field if requested
  //////////////////////////////////////////////////////////////////////
  // EstimateBiasField(clo,10.0,1.0,sm);
  // std::shared_ptr<const NEWIMAGE::volume<float> > bfield = sm.GetBiasField();
  // exit(EXIT_SUCCESS);

  // Set initial slice-to-vol parameters. This option is only for testing/debugging/personal use
  if (clo.IsSliceToVol() && clo.InitS2VFname() != std::string("")) {
    sm.SetMovementModelOrder(clo.MovementModelOrder(0));
    if (clo.RegisterDWI() && clo.Registerb0()) sm.SetS2VMovement(clo.InitS2VFname(),ANY);
    else if (clo.RegisterDWI()) sm.SetS2VMovement(clo.InitS2VFname(),DWI);
    else sm.SetS2VMovement(clo.InitS2VFname(),B0);
  }

  //////////////////////////////////////////////////////////////////////
  // Next do the slice-to-vol registration if requested.
  //////////////////////////////////////////////////////////////////////
  if (clo.IsSliceToVol() && clo.S2V_NIter(0) > 0 && clo.InitS2VFname() == std::string("")) {
    double s2v_key = prof.StartEntry("Calling DoSliceToVolumeRegistration");
    if (clo.Verbose()) cout << "Performing slice-to-volume registration" << endl;
    if (clo.EstimateMoveBySusc()) { // Restrict EC if we are to eventually estimate MBS
      EDDY::SecondLevelECModel b0_slm = clo.b0_SecondLevelModel();
      EDDY::SecondLevelECModel dwi_slm = clo.SecondLevelModel();
      clo.Set_b0_SecondLevelModel(EDDY::Linear_2nd_lvl_mdl);
      clo.SetSecondLevelModel(EDDY::Linear_2nd_lvl_mdl);
      for (unsigned int i=0; i<clo.NumOfNonZeroMovementModelOrder(); i++) {
        if (clo.Verbose()) cout << "Setting slice-to-volume order to " << clo.MovementModelOrder(i) << endl;
        sm.SetMovementModelOrder(clo.MovementModelOrder(i));
        sm.Set_S2V_Lambda(clo.S2V_Lambda(i));
        dwi_rm = DoSliceToVolumeRegistration(clo,i,false,sm,dwi_rm);
        sm.ApplyLocationReference();
      }
      clo.Set_b0_SecondLevelModel(b0_slm);
      clo.SetSecondLevelModel(dwi_slm);
    }
    else {
      for (unsigned int i=0; i<clo.NumOfNonZeroMovementModelOrder(); i++) {
        if (clo.Verbose()) cout << "Setting slice-to-volume order to " << clo.MovementModelOrder(i) << endl;
        sm.SetMovementModelOrder(clo.MovementModelOrder(i));
        sm.Set_S2V_Lambda(clo.S2V_Lambda(i));
        dwi_rm = DoSliceToVolumeRegistration(clo,i,false,sm,dwi_rm);
        sm.ApplyLocationReference();
      }
    }
    prof.EndEntry(s2v_key);

    // Do another check for residual position differences in case s2v wrecked things
    if (clo.RegisterDWI()) {
      double peas_key = prof.StartEntry("Calling PostEddyAlignShells after S2V");
      if (!clo.SeparateOffsetFromMovement() && !clo.AlignShellsPostEddy()) {
        if (clo.Verbose()) cout << "Checking shell alignment along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
        PEASUtils::PostEddyAlignShellsAlongPE(clo,false,sm);
        if (clo.Verbose()) cout << "Checking shell alignment (running PostEddyAlignShells)" << endl;
        PEASUtils::PostEddyAlignShells(clo,false,sm);
      }
      else if (clo.SeparateOffsetFromMovement() && !clo.AlignShellsPostEddy()) {
        if (clo.Verbose()) cout << "Aligning shells along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
        PEASUtils::PostEddyAlignShellsAlongPE(clo,true,sm);
        if (clo.Verbose()) cout << "Checking shell alignment (running PostEddyAlignShells)" << endl;
        PEASUtils::PostEddyAlignShells(clo,false,sm);
      }
      else if (clo.AlignShellsPostEddy()) {
        if (clo.Verbose()) cout << "Checking shell alignment along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
        PEASUtils::PostEddyAlignShellsAlongPE(clo,false,sm);
        if (clo.Verbose()) cout << "Aligning shells (running PostEddyAlignShells)" << endl;
        PEASUtils::PostEddyAlignShells(clo,true,sm);
      }
      prof.EndEntry(peas_key);
    }
  }

  // Set initial MBS derivative fields. This option is only for testing/debugging/personal use
  if (clo.EstimateMoveBySusc() && clo.InitMBSFname() != std::string("")) {
    NEWIMAGE::volume4D<float> mbs_init;
    NEWIMAGE::read_volume4D(mbs_init,clo.InitMBSFname());
    for (unsigned int i=0; i<clo.MoveBySuscParam().size(); i++) {
      sm.SetDerivSuscField(clo.MoveBySuscParam()[i],mbs_init[i]);
    }
  }

  //////////////////////////////////////////////////////////////////////
  // Estimate movement-by-susceptibility interaction if requested.
  //////////////////////////////////////////////////////////////////////
  if (clo.EstimateMoveBySusc() && clo.MoveBySuscNiter()>0) {
    sm.SetUseB0sToInformDWIRegistration(false); // So as to preserve current estimates
    std::vector<unsigned int> b0s;
    std::vector<unsigned int> dwis;
    if (clo.IsSliceToVol()) {           // Use only "steady" volumes if we know who they are
      EDDY::s2vQuant s2vq(sm,1.0,1.0);  // "Steadiness hardcoded to 1mm and 1degree
      if (clo.Registerb0()) b0s = s2vq.FindStillVolumes(B0,clo.MoveBySuscParam());
      if (clo.RegisterDWI()) dwis = s2vq.FindStillVolumes(DWI,clo.MoveBySuscParam());
    }
    else { // Otherwise use all volumes
      if (clo.Registerb0()) { b0s.resize(sm.NScans(B0)); for (unsigned int i=0; i<sm.NScans(B0); i++) b0s[i] = i; }
      if (clo.RegisterDWI()) { dwis.resize(sm.NScans(DWI)); for (unsigned int i=0; i<sm.NScans(DWI); i++) dwis[i] = i; }
    }
    if (clo.RegisterDWI()) { // Do interleaved MBS and EC/movement estimation
      unsigned int mbs_niter = (clo.MoveBySuscNiter() / clo.N_MBS_Interleaves()) + 1;
      unsigned int niter, s2vi=0;
      if (clo.IsSliceToVol()) {
        s2vi = clo.NumOfNonZeroMovementModelOrder()-1;
        niter = clo.S2V_NIter(s2vi);
        sm.SetMovementModelOrder(clo.MovementModelOrder(s2vi));
        sm.Set_S2V_Lambda(clo.S2V_Lambda(s2vi));
        clo.SetS2VParam(clo.MovementModelOrder(s2vi),clo.S2V_Lambda(s2vi),0.0,(niter/clo.N_MBS_Interleaves())+1);
      }
      else { niter = clo.NIter(); clo.SetNIterAndFWHM((niter/clo.N_MBS_Interleaves())+1,std::vector<float>(1,0.0)); }
      NEWMAT::ColumnVector spar;
      EDDY::MoveBySuscCF cf(sm,clo,b0s,dwis,clo.MoveBySuscParam(),clo.MoveBySuscOrder(),clo.MoveBySuscKsp());

      for (unsigned int i=0; i<clo.N_MBS_Interleaves(); i++) {
        if (clo.Verbose()) cout << "Running interleave " << i+1 << " of MBS" << endl;
        if (!i) spar = cf.Par(); // Start guesses all zeros for first iteration
        cf.SetLambda(clo.MoveBySuscLambda());
        MISCMATHS::NonlinParam nlp(cf.NPar(),MISCMATHS::NL_LM,spar);
        nlp.SetMaxIter(mbs_niter);
        nlp.SetGaussNewtonType(MISCMATHS::LM_GN);
        double mbs_key = prof.StartEntry("Calling nonlin for MBS");
        MISCMATHS::nonlin(nlp,cf);
        prof.EndEntry(mbs_key);
        spar = cf.Par(); // Save for next iteration
        if (clo.IsSliceToVol()) {
          if (clo.Verbose()) cout << "Running slice-to-vol interleaved with MBS" << endl;
          double s2v_key = prof.StartEntry("Calling DoSliceToVolumeRegistration as part of MBS");
          dwi_rm = DoSliceToVolumeRegistration(clo,s2vi,false,sm,dwi_rm);
          sm.ApplyLocationReference();
          prof.EndEntry(s2v_key);
        }
        else {
          if (clo.Verbose()) cout << "Running vol-to-vol interleaved with MBS" << endl;
          double v2v_key = prof.StartEntry("Calling DoVolumeToVolumeRegistration as part of MBS");
          dwi_rm = DoVolumeToVolumeRegistration(clo,sm);
          sm.ApplyLocationReference();
          prof.EndEntry(v2v_key);
        }
      }
      cf.WriteFirstOrderFields(clo.MoveBySuscFirstOrderFname());
      if (clo.MoveBySuscOrder() > 1) cf.WriteSecondOrderFields(clo.MoveBySuscSecondOrderFname());
    }
    else { // Just do a straightforward MBS estimation
      // Make cost-function object for movement-by-susceptibility
      EDDY::MoveBySuscCF cf(sm,clo,b0s,dwis,clo.MoveBySuscParam(),clo.MoveBySuscOrder(),clo.MoveBySuscKsp());
      NEWMAT::ColumnVector spar = cf.Par(); // Start guesses (all zeros);
      cf.SetLambda(clo.MoveBySuscLambda());
      MISCMATHS::NonlinParam nlp(cf.NPar(),MISCMATHS::NL_LM,spar);
      nlp.SetMaxIter(clo.MoveBySuscNiter());
      nlp.SetGaussNewtonType(MISCMATHS::LM_GN);
      double mbs_key = prof.StartEntry("Calling nonlin for MBS");
      MISCMATHS::nonlin(nlp,cf);            // Estimate
      prof.EndEntry(mbs_key);
      cf.WriteFirstOrderFields(clo.MoveBySuscFirstOrderFname());
      if (clo.MoveBySuscOrder() > 1) cf.WriteSecondOrderFields(clo.MoveBySuscSecondOrderFname());
    }

    // Do another check for residual position differences in case MBS wrecked things
    if (clo.RegisterDWI()) {
      double peas_key = prof.StartEntry("Calling PostEddyAlignShells after MBS");
      if (!clo.SeparateOffsetFromMovement() && !clo.AlignShellsPostEddy()) {
        if (clo.Verbose()) cout << "Checking shell alignment along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
        PEASUtils::PostEddyAlignShellsAlongPE(clo,false,sm);
        if (clo.Verbose()) cout << "Checking shell alignment (running PostEddyAlignShells)" << endl;
        PEASUtils::PostEddyAlignShells(clo,false,sm);
      }
      else if (clo.SeparateOffsetFromMovement() && !clo.AlignShellsPostEddy()) {
        if (clo.Verbose()) cout << "Aligning shells along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
        PEASUtils::PostEddyAlignShellsAlongPE(clo,true,sm);
        if (clo.Verbose()) cout << "Checking shell alignment (running PostEddyAlignShells)" << endl;
        PEASUtils::PostEddyAlignShells(clo,false,sm);
      }
      else if (clo.AlignShellsPostEddy()) {
        if (clo.Verbose()) cout << "Checking shell alignment along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
        PEASUtils::PostEddyAlignShellsAlongPE(clo,false,sm);
        if (clo.Verbose()) cout << "Aligning shells (running PostEddyAlignShells)" << endl;
        PEASUtils::PostEddyAlignShells(clo,true,sm);
      }
      prof.EndEntry(peas_key);
    }
  }

  //////////////////////////////////////////////////////////////////////
  // Do final check (and possibly replacement) of outliers with ff=1
  // (very little Q-space smoothing).
  //////////////////////////////////////////////////////////////////////
  if (clo.RegisterDWI()) {
    if (clo.Verbose()) cout << "Performing final outlier check" << endl;
    double old_hypar_ff = 1.0;
    if (clo.HyParFudgeFactor() != 1.0) { old_hypar_ff = clo.HyParFudgeFactor(); clo.SetHyParFudgeFactor(1.0); }
    double folc_key = prof.StartEntry("Calling FinalOLCheck");
    dwi_rm = FinalOLCheck(clo,dwi_rm,sm);
    prof.EndEntry(folc_key);
    if (old_hypar_ff != 1.0) clo.SetHyParFudgeFactor(old_hypar_ff);
    // Write outlier information
    double wol_key  = prof.StartEntry("Writing outlier information");
    std::vector<unsigned int> i2i = sm.GetDwi2GlobalIndexMapping();
    dwi_rm->WriteReport(i2i,clo.OLReportFname());
    dwi_rm->WriteMatrixReport(i2i,sm.NScans(),clo.OLMapReportFname(),clo.OLNStDevMapReportFname(),clo.OLNSqrStDevMapReportFname());
    if (clo.WriteOutlierFreeData()) {
      if (clo.Verbose()) cout << "Running sm.WriteOutlierFreeData" << endl;
      sm.WriteOutlierFreeData(clo.OLFreeDataFname());
    }
    prof.EndEntry(wol_key);
  }

  // Add rotation. Hidden function. ONLY to be used to
  // test that rotation of b-vecs does the right thing.
  if (clo.DoTestRot()) {
    if (clo.Verbose()) cout << "Running sm.AddRotation" << endl;
    sm.AddRotation(clo.TestRotAngles());
  }

  // Write registration parameters
  if (clo.Verbose()) cout << "Running sm.WriteParameterFile" << endl;
  if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteParameterFile(clo.ParOutFname());
  else if (clo.RegisterDWI()) sm.WriteParameterFile(clo.ParOutFname(),DWI);
  else sm.WriteParameterFile(clo.ParOutFname(),B0);

  // Write movement-over-time file if SliceToVol registration was performed
  if (sm.IsSliceToVol()) {
    if (clo.Verbose()) cout << "Running sm.WriteMovementOverTimeFile" << endl;
    if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteMovementOverTimeFile(clo.MovementOverTimeOutFname());
    else if (clo.RegisterDWI()) sm.WriteMovementOverTimeFile(clo.MovementOverTimeOutFname(),DWI);
    else sm.WriteMovementOverTimeFile(clo.MovementOverTimeOutFname(),B0);
  }

  // Write registered images
  if (clo.Verbose()) cout << "Running sm.WriteRegisteredImages" << endl;
  double wri_key = prof.StartEntry("Writing registered images");
  if (!clo.ReplaceOutliers()) { if (clo.Verbose()) { cout << "Running sm.RecycleOutliers" << endl; } sm.RecycleOutliers(); } // Bring back original data
  if (sm.IsSliceToVol()) { // If we need to get predictions to support the resampling
    NEWIMAGE::volume4D<float> pred;
    ScanType st;
    if (clo.RegisterDWI() && clo.Registerb0()) st=ANY; else if (clo.RegisterDWI()) st=DWI; else st=B0;
    GetPredictionsForResampling(clo,st,sm,pred);
    // Set resampling in slice direction to spline.
    EDDY::PolationPara old_pp = sm.GetPolation();
    EDDY::PolationPara new_pp = old_pp;
    if (old_pp.GetS2VInterp() != NEWIMAGE::spline) { new_pp.SetS2VInterp(NEWIMAGE::spline); sm.SetPolation(new_pp); }
    if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteRegisteredImages(clo.IOutFname(),clo.OutMaskFname(),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),pred);
    else if (clo.RegisterDWI()) sm.WriteRegisteredImages(clo.IOutFname(),clo.OutMaskFname(),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),pred,DWI);
    else sm.WriteRegisteredImages(clo.IOutFname(),clo.OutMaskFname(),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),pred,B0);
    if (old_pp.GetS2VInterp() != NEWIMAGE::spline) sm.SetPolation(old_pp);
  }
  else {
    if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteRegisteredImages(clo.IOutFname(),clo.OutMaskFname(),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput());
    else if (clo.RegisterDWI()) sm.WriteRegisteredImages(clo.IOutFname(),clo.OutMaskFname(),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),DWI);
    else sm.WriteRegisteredImages(clo.IOutFname(),clo.OutMaskFname(),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),B0);
  }
  prof.EndEntry(wri_key);

  // Optionally write out a "data set" that consists of the GP predictions. This
  // was added to demonstrate to Kings that it makes b***er all difference.
  if (clo.WritePredictions() || clo.WriteScatterBrainPredictions()) {
    // First write predictions in model space
    NEWIMAGE::volume4D<float> pred;
    EDDY::ScanType st;
    std::vector<double> hypar;
    if (clo.RegisterDWI() && clo.Registerb0()) st=ANY; else if (clo.RegisterDWI()) st=DWI; else st=B0;
    if (clo.WritePredictions()) { // Do the "regular" fitting to the GP, but ensure spline interpolation in z if s2v
      if (clo.Verbose()) cout << "Running EDDY::GetPredictionsForResampling" << endl;
      double wpred_key = prof.StartEntry("Calculating and writing predictions");
      EDDY::PolationPara old_pol = sm.GetPolation();
      EDDY::PolationPara new_pol = old_pol;
      if (clo.IsSliceToVol() && new_pol.GetS2VInterp() != NEWIMAGE::spline) new_pol.SetS2VInterp(NEWIMAGE::spline);
      sm.SetPolation(new_pol);
      EddyCommandLineOptions tmp_clo = clo;
      if (!tmp_clo.RotateBVecsDuringEstimation()) tmp_clo.SetRotateBVecsDuringEstimation(true);
      hypar = GetPredictionsForResampling(tmp_clo,st,sm,pred);
      pred /= sm.ScaleFactor();
      NEWIMAGE::write_volume(pred,clo.PredictionsOutFname());
      sm.SetPolation(old_pol);
      // Next write predictions in scan space(s)
      pred = 0.0;
      hypar = GetPredictionsForResampling(clo,st,sm,pred);
      pred /= sm.ScaleFactor();
      for (int s=0; s<pred.tsize(); s++) {
        pred[s] = EddyUtils::TransformModelToScanSpace(pred[s],sm.Scan(s,st),sm.GetSuscHzOffResField(s,st));
      }
      NEWIMAGE::write_volume(pred,clo.PredictionsInScanSpaceOutFname());
      prof.EndEntry(wpred_key);
    }
    if (clo.WriteScatterBrainPredictions()) { // Write predictions using "scattered data approach"
      if (clo.Verbose()) cout << "Running EDDY::GetScatterBrainPredictions" << endl;
      double wspred_key = prof.StartEntry("Calculating and writing scatter brain predictions");
      GetScatterBrainPredictions(clo,st,sm,hypar,pred);
      pred /= sm.ScaleFactor();
      NEWIMAGE::write_volume(pred,clo.ScatterBrainPredictionsOutFname());
      prof.EndEntry(wspred_key);
    }
    if (clo.WriteVolumeScatterBrainPredictions()) { // Write predictions using "scattered data approach" but with volume based bvec rotation
      if (clo.Verbose()) cout << "Running EDDY::GetScatterBrainPredictions with volume based bvec rotation" << endl;
      double wspred_key = prof.StartEntry("Calculating and writing scatter brain predictions");
      GetScatterBrainPredictions(clo,st,sm,hypar,pred,true);
      pred /= sm.ScaleFactor();
      NEWIMAGE::write_volume(pred,clo.VolumeScatterBrainPredictionsOutFname());
      prof.EndEntry(wspred_key);
    }
  }

  // Optionally write an additional set of registered images, this time with outliers retained.
  // This was added for the benefit of the HCP.
  if (clo.ReplaceOutliers() && clo.WriteAdditionalResultsWithOutliersRetained()) {
    double wri_key = prof.StartEntry("Writing registered images with outliers retained");
    if (clo.Verbose()) cout << "Running sm.WriteRegisteredImages" << endl;
    if (clo.Verbose()) { cout << "Running sm.RecycleOutliers" << endl; }
    sm.RecycleOutliers(); // Bring back original data
    if (sm.IsSliceToVol()) { // If we need to get predictions to support the resampling
      NEWIMAGE::volume4D<float> pred;
      ScanType st;
      if (clo.RegisterDWI() && clo.Registerb0()) st=ANY; else if (clo.RegisterDWI()) st=DWI; else st=B0;
      GetPredictionsForResampling(clo,st,sm,pred);
      if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteRegisteredImages(clo.AdditionalWithOutliersOutFname(),std::string(""),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),pred);
      else if (clo.RegisterDWI()) sm.WriteRegisteredImages(clo.AdditionalWithOutliersOutFname(),std::string(""),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),pred,DWI);
      else sm.WriteRegisteredImages(clo.AdditionalWithOutliersOutFname(),std::string(""),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),pred,B0);
    }
    else {
      if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteRegisteredImages(clo.AdditionalWithOutliersOutFname(),std::string(""),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput());
      else if (clo.RegisterDWI()) sm.WriteRegisteredImages(clo.AdditionalWithOutliersOutFname(),std::string(""),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),DWI);
      else sm.WriteRegisteredImages(clo.AdditionalWithOutliersOutFname(),std::string(""),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),B0);
    }
    prof.EndEntry(wri_key);
  }

  // Write EC fields
  if (clo.WriteFields()) {
    if (clo.Verbose()) cout << "Running sm.WriteECFields" << endl;
    if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteECFields(clo.ECFOutFname());
    else if (clo.RegisterDWI()) sm.WriteECFields(clo.ECFOutFname(),DWI);
    else sm.WriteECFields(clo.ECFOutFname(),B0);
  }

  // Write rotated b-vecs
  if (clo.WriteRotatedBVecs()) {
    if (clo.Verbose()) cout << "Running sm.WriteRotatedBVecs" << endl;
    sm.WriteRotatedBVecs(clo.RotatedBVecsOutFname());
  }

  // Write movement RMS
  if (clo.WriteMovementRMS()) {
    double rms_key = prof.StartEntry("Writing RMS");
    if (clo.Verbose()) cout << "Running sm.WriteMovementRMS" << endl;
    if (clo.RegisterDWI() && clo.Registerb0()) { sm.WriteMovementRMS(clo.RMSOutFname()); sm.WriteRestrictedMovementRMS(clo.RestrictedRMSOutFname()); }
    else if (clo.RegisterDWI()) { sm.WriteMovementRMS(clo.RMSOutFname(),DWI); sm.WriteRestrictedMovementRMS(clo.RestrictedRMSOutFname(),DWI); }
    else { sm.WriteMovementRMS(clo.RMSOutFname(),B0); sm.WriteRestrictedMovementRMS(clo.RestrictedRMSOutFname(),B0); }
    prof.EndEntry(rms_key);
  }

  // Write CNR maps
  if (clo.WriteCNRMaps() || clo.WriteRangeCNRMaps() || clo.WriteResiduals()) {
    double cnr_key = prof.StartEntry("Writing CNR maps");
    double old_hypar_ff = 1.0;
    if (clo.HyParFudgeFactor() != 1.0) { old_hypar_ff = clo.HyParFudgeFactor(); clo.SetHyParFudgeFactor(1.0); }
    if (clo.Verbose()) cout << "Running EDDY::WriteCNRMaps" << endl;
    WriteCNRMaps(clo,sm,clo.CNROutFname(),clo.RangeCNROutFname(),clo.ResidualsOutFname());
    if (old_hypar_ff != 1.0) clo.SetHyParFudgeFactor(old_hypar_ff);
    prof.EndEntry(cnr_key);
  }

  // Write 3D displacement fields
  if (clo.WriteDisplacementFields()) {
    if (clo.Verbose()) cout << "Running sm.WriteDisplacementFields" << endl;
    if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteDisplacementFields(clo.DFieldOutFname());
    else if (clo.RegisterDWI()) sm.WriteDisplacementFields(clo.DFieldOutFname(),DWI);
    else sm.WriteDisplacementFields(clo.DFieldOutFname(),B0);
  }

  prof.EndEntry(total_key);
  return(EXIT_SUCCESS);
}
catch(const std::exception& e)
{
  cout << "EDDY::: Eddy failed with message " << e.what() << endl;
  return(EXIT_FAILURE);
}
catch(...)
{
  cout << "EDDY::: Eddy failed" << endl;
  return(EXIT_FAILURE);
}

namespace EDDY {

/****************************************************************//**
*
*  A very high-level global function that registers all the scans
*  (b0 and dwis) in sm using a volume-to-volume model.
*  \param[in] clo Carries information about the command line options
*  that eddy was invoked with.
*  \param[in,out] sm Collection of all scans. Will be updated by this call.
*  \return A ptr to ReplacementManager that details which slices in
*  which dwi scans were replaced by their expectations.
*
********************************************************************/
ReplacementManager *DoVolumeToVolumeRegistration(// Input
                                                 const EddyCommandLineOptions&  clo,
                                                 // Input/Output
                                                 ECScanManager&                 sm) EddyTry
{
  static Utilities::FSLProfiler prof("_"+string(__FILE__)+"_"+string(__func__));
  double total_key = prof.StartEntry("Total");
  // Start by registering the b0 scans
  NEWMAT::Matrix b0_mss, b0_ph;
  ReplacementManager *b0_rm = NULL;
  if (clo.NIter() && clo.Registerb0() && sm.NScans(B0)>1) {
    double b0_key = prof.StartEntry("b0");
    if (clo.Verbose()) cout << "Running Register" << endl;
    b0_rm = Register(clo,B0,clo.NIter(),clo.FWHM(),clo.b0_SecondLevelModel(),false,sm,b0_rm,b0_mss,b0_ph);
    if (clo.IsSliceToVol()) { // Find best scan for shape reference if we are to do also slice-to-volume registration
      double minmss=1e20;
      unsigned int mindx=0;
      for (unsigned int i=0; i<sm.NScans(B0); i++) {
        if (b0_mss(b0_mss.Nrows(),i+1) < minmss) { minmss=b0_mss(b0_mss.Nrows(),i+1); mindx=i; }
      }
      if (clo.Verbose()) cout << "Setting scan " << sm.Getb02GlobalIndexMapping(mindx) << " as b0 shape-reference."<< endl;
      sm.SetB0ShapeReference(sm.Getb02GlobalIndexMapping(mindx));
    }
    // Apply reference for location
    if (clo.Verbose()) cout << "Running sm.ApplyB0LocationReference" << endl;
    sm.ApplyB0LocationReference();
    prof.EndEntry(b0_key);
  }
  // See if we can use b0 movement estimates to inform dwi registration
  if (sm.B0sAreInterspersed() && sm.UseB0sToInformDWIRegistration() && clo.Registerb0() && clo.RegisterDWI()) {
    if (clo.Verbose()) cout << "Running sm.PolateB0MovPar" << endl;
    sm.PolateB0MovPar();
  }
  // Now register the dwi scans
  NEWMAT::Matrix dwi_mss, dwi_ph;
  ReplacementManager *dwi_rm = NULL;
  if (clo.NIter() && clo.RegisterDWI()) {
    double dwi_key = prof.StartEntry("dwi");
    if (clo.Verbose()) cout << "Running Register" << endl;
    dwi_rm = Register(clo,DWI,clo.NIter(),clo.FWHM(),clo.SecondLevelModel(),true,sm,dwi_rm,dwi_mss,dwi_ph);
    if (clo.IsSliceToVol()) { // Find best scan for shape reference if we are to do also slice-to-volume registration
      std::vector<double> bvals;
      std::vector<std::vector<unsigned int> > shindx = sm.GetShellIndicies(bvals);
      for (unsigned int shell=0; shell<shindx.size(); shell++) {
        double minmss=1e20;
        unsigned int mindx=0;
        bool found_vol_with_no_outliers=false;
        for (unsigned int i=0; i<shindx[shell].size(); i++) {
          if (!sm.Scan(shindx[shell][i]).HasOutliers()) { // Only consider scans without outliers
            found_vol_with_no_outliers=true;
            if (dwi_mss(dwi_mss.Nrows(),sm.GetGlobal2DWIIndexMapping(shindx[shell][i])+1) < minmss) {
              minmss=dwi_mss(dwi_mss.Nrows(),sm.GetGlobal2DWIIndexMapping(shindx[shell][i])+1);
              mindx=shindx[shell][i];
            }
          }
        }
        if (!found_vol_with_no_outliers) {
          std::vector<unsigned int> i2i = sm.GetDwi2GlobalIndexMapping();
          dwi_rm->WriteReport(i2i,clo.OLReportFname());
          dwi_rm->WriteMatrixReport(i2i,sm.NScans(),clo.OLMapReportFname(true),clo.OLNStDevMapReportFname(true),clo.OLNSqrStDevMapReportFname(true));
          std::ostringstream errtxt;
          errtxt << "DoVolumeToVolumeRegistration: Unable to find volume with no outliers in shell " << shell << " with b-value=" << bvals[shell];
          throw EddyException(errtxt.str());
        }
        if (clo.Verbose()) cout << "Setting scan " << mindx << " as shell shape-reference for shell "<< shell << " with b-value= " << bvals[shell] << endl;
        sm.SetShellShapeReference(shell,mindx);
      }
      // Apply reference for location
    }
    if (clo.Verbose()) cout << "Running sm.ApplyDWILocationReference" << endl;
    sm.ApplyDWILocationReference();
    prof.EndEntry(dwi_key);
  }
  // Write history of cost-function and parameter estimates if requested
  if (clo.NIter() && clo.History()) {
    if (clo.RegisterDWI()) {
      MISCMATHS::write_ascii_matrix(clo.DwiMssHistoryFname(),dwi_mss);
      MISCMATHS::write_ascii_matrix(clo.DwiParHistoryFname(),dwi_ph);
    }
    if (clo.Registerb0()) {
      MISCMATHS::write_ascii_matrix(clo.B0MssHistoryFname(),b0_mss);
      MISCMATHS::write_ascii_matrix(clo.B0ParHistoryFname(),b0_ph);
    }
  }
  prof.EndEntry(total_key);
  return(dwi_rm);
} EddyCatch

/****************************************************************//**
*
*  A very high-level global function that registers all the scans
*  (b0 and dwis) in sm using a slice-to-volume model.
*  \param[in] clo Carries information about the command line options
*  that eddy was invoked with.
*  \param[in,out] sm Collection of all scans. Will be updated by this call.
*  \param[out] msshist Returns the history of the mss. msshist(i,j)
*  contains the mss for the jth scan on the ith iteration.
*  \param[out] phist Returns the history of the estimated parameters.
*  phist(i,j) contains the jth parameter on the ith iteration
*  \return A ptr to ReplacementManager that details which slices in
*  which dwi scans were replaced by their expectations.
*
********************************************************************/

ReplacementManager *DoSliceToVolumeRegistration(// Input
                                                const EddyCommandLineOptions&  clo,
                                                unsigned int                   oi,        // Order index
                                                bool                           dol,       // Detect outliers?
                                                // Input/Output
                                                ECScanManager&                 sm,
                                                ReplacementManager             *dwi_rm) EddyTry
{
  static Utilities::FSLProfiler prof("_"+string(__FILE__)+"_"+string(__func__));
  double total_key = prof.StartEntry("Total");
  // Start by registering the b0 scans
  NEWMAT::Matrix b0_mss, b0_ph;
  ReplacementManager *b0_rm = NULL;
  if (clo.S2V_NIter(oi) && clo.Registerb0() && sm.NScans(B0)>1) {
    double b0_key = prof.StartEntry("b0");
    if (clo.Verbose()) cout << "Running Register" << endl;
    b0_rm = Register(clo,B0,clo.S2V_NIter(oi),clo.S2V_FWHM(oi),clo.b0_SecondLevelModel(),false,sm,b0_rm,b0_mss,b0_ph);
    // Set reference for shape
    if (clo.Verbose()) cout << "Running sm.ApplyB0ShapeReference" << endl;
    sm.ApplyB0ShapeReference();
    // Set reference for location
    if (clo.Verbose()) cout << "Running sm.ApplyB0LocationReference" << endl;
    sm.ApplyB0LocationReference();
    prof.EndEntry(b0_key);
  }
  // Now register the dwi scans
  NEWMAT::Matrix dwi_mss, dwi_ph;
  if (clo.S2V_NIter(oi) && clo.RegisterDWI()) {
    double dwi_key = prof.StartEntry("dwi");
    if (clo.Verbose()) cout << "Running Register" << endl;
    dwi_rm = Register(clo,DWI,clo.S2V_NIter(oi),clo.S2V_FWHM(oi),clo.SecondLevelModel(),dol,sm,dwi_rm,dwi_mss,dwi_ph);
    // Set reference for shape
    if (clo.Verbose()) cout << "Running sm.ApplyShellShapeReference" << endl;
    for (unsigned int si=0; si<sm.NoOfShells(DWI); si++) sm.ApplyShellShapeReference(si);
    // Set reference for location
    if (clo.Verbose()) cout << "Running sm.ApplyDWILocationReference" << endl;
    sm.ApplyDWILocationReference();
    prof.EndEntry(dwi_key);
  }
  // Write history of cost-function and parameter estimates if requested
  if (clo.S2V_NIter(oi) && clo.History()) {
    if (clo.RegisterDWI()) {
      MISCMATHS::write_ascii_matrix(clo.DwiMssS2VHistoryFname(),dwi_mss);
      MISCMATHS::write_ascii_matrix(clo.DwiParS2VHistoryFname(),dwi_ph);
    }
    if (clo.Registerb0()) {
      MISCMATHS::write_ascii_matrix(clo.B0MssS2VHistoryFname(),b0_mss);
      MISCMATHS::write_ascii_matrix(clo.B0ParS2VHistoryFname(),b0_ph);
    }
  }
  prof.EndEntry(total_key);
  return(dwi_rm);
} EddyCatch

/****************************************************************//**
*
*  A very high-level global function that estimates a receieve
*  bias field that is stationary in the scanner framework.
*  \param[in] clo Carries information about the command line options
*  that eddy was invoked with.
*  \param[in] ksp Knotspacing of spline representation of field.
*  Set to negative value for free-form field.
*  \param[in] lambda Weight of regularisation when estimating field.
*  \param[in,out] sm Collection of all scans. Will be updated by this call.
*
********************************************************************/
/*
void EstimateBiasField(// Input
                       const EddyCommandLineOptions&  clo,
                       double                         ksp,
                       double                         lambda,
                       // Input/output
                       ECScanManager&                 sm) EddyTry
{
  BiasFieldEstimator bfe;
  ScanType st[2] = {B0, DWI};
  NEWIMAGE::volume<float> mask = sm.Scan(0,ANY).GetIma(); EddyUtils::SetTrilinearInterp(mask); mask = 1.0; // FOV-mask in model space

  #ifdef COMPILE_GPU
  // Loop over B0s and DWIs
  for (unsigned int sti=0; sti<2; sti++) {
    if (sm.NScans(st[sti]) > 1) { // Will only have info on field if 2 or more locations
      std::shared_ptr<DWIPredictionMaker> pmp = EddyGpuUtils::LoadPredictionMaker(clo,st[sti],sm,0,0.0,mask);
      // First calculate the average location for this scan type to use as reference
      EDDY::ImageCoordinates mic = sm.Scan(0,st[sti]).SamplingPoints();
      for (unsigned int s=1; s<sm.NScans(st[sti]); s++) {
        mic += sm.Scan(s,st[sti]).SamplingPoints();
      }
      mic /= static_cast<float>(sm.NScans(st[sti]));
      bfe.SetRefScan(mask,mic);
      // Next loop over chunks of predictions and add scans
      for (GpuPredictorChunk c(sm.NScans(st[sti]),mask); c<sm.NScans(st[sti]); c++) {
        std::vector<unsigned int> si = c.Indicies();
        std::vector<NEWIMAGE::volume<float> > pred = pmp->Predict(si);
        // Loop over scans within chunk
        for (unsigned int i=0; i<si.size(); i++) {
          // EDDY::ImageCoordinates ic = EddyGpuUtils::GetCoordinatesInScannerSpace(sm.Scan(si[i],st[sti]));
          // cout << "i = " << i << ", calling SamplingPoints()" << endl;
          EDDY::ImageCoordinates ic = sm.Scan(si[i],st[sti]).SamplingPoints();
          bfe.AddScan(pred[i],EddyGpuUtils::GetUnwarpedScan(sm.Scan(si[i],st[sti]),sm.GetSuscHzOffResField(si[i],st[sti]),true,sm.GetPolation()),mask,ic);
        }
      }
    }
  }
  #else
  // Loop over B0s and DWIs
  for (unsigned int sti=0; sti<2; sti++) {
    std::shared_ptr<DWIPredictionMaker> pmp = EDDY::LoadPredictionMaker(clo,st[sti],sm,0,0.0,mask);
    // Loop over scans
    #pragma omp parallel for
    for (int s=0; s<int(sm.NScans(st[sti])); s++) {
      // Get coordinates in scanner space
      EDDY::ImageCoordinates ic = sm.Scan(s,st[sti]).SamplingPoints();
      // Get prediction in model space
      NEWIMAGE::volume<float> pred = pmp->Predict(s);
      // Get original data in model space
      NEWIMAGE::volume<float> sm.Scan(s,st[sti]).GetUnwarpedIma(sm.GetSuscHzOffResField(s,st[sti]));
      // AddScan
      bfe.AddScan(pred,sm.Scan(s,st[sti]).GetUnwarpedIma(sm.GetSuscHzOffResField(s,st[sti])),mask,ic);
    }
  }
  #endif
  // Caclulate many fields
  // double iksp[] = {10.0};
  // double ilambda[] = {1e-10, 1e-6, 1e-2, 100, 1e6, 1e10, 1e14};
  // for (unsigned int i=0; i<7; i++) {
    // cout << "Calculating field with lambda = " << ilambda[i] << endl;
    // NEWIMAGE::volume<float> bfield = bfe.GetField(ilambda[i]);
    // char fname[256]; sprintf(fname,"bfield_%d",i);
    // NEWIMAGE::write_volume(bfield,fname);
  // }

  // Calculate field
  NEWIMAGE::volume<float> bfield = bfe.GetField(1e10);
  NEWIMAGE::write_volume(bfield,"bfield_1e10");

  // Set field
  sm.SetBiasField(bfield);

  // Do a second step where the unknown offset is calculated by
  // maximising the SNR/CNR of the corrected data.

  // float offset = EstimateBiasFieldOffset();

  return;
} EddyCatch
*/

/****************************************************************//**
*
*  A very high-level global function that estimates the offset
*  of an estimated bias field. It does that by finding the offset
*  that maximises a weighted sum of the SNR (b0-volumes) and CNR
*  (dwi-volumes) of the corrected data. It will return the estimated
*  offset, and also set it in the ScanManager. It will also set the
*  average of the bias field to one within the user defined mask.
*  \returns offset The estimated offset value
*  \param[in] clo Carries information about the command line options
*  that eddy was invoked with.
*  \param[in,out] sm Collection of all scans. Will be updated by this call.
*
********************************************************************/
float EstimatedBiasFieldOffset(// Input
                               const EddyCommandLineOptions&  clo,
                               // Input/output
                               ECScanManager&                 sm) EddyTry
{
  return(1.0);
} EddyCatch

/****************************************************************//**
*
*  A global function that registers the scans in sm.
*  \param[in] clo Carries information about the command line options
*  that eddy was invoked with.
*  \param[in] st Specifies if we should register the diffusion weighted
*  images or the b=0 images.
*  \param[in] niter Specifies how many iterations should be run
*  \param[in] slm Specifies if a 2nd level model should be used to
*  constrain the estimates.
*  \param[in] dol Detect outliers if true
*  \param[in,out] sm Collection of all scans. Will be updated by this call.
*  \param[in,out] rm Pointer to ReplacementManager. If NULL on input
*  one will be allocated and the new pointer value passed on return.
*  \param[out] msshist Returns the history of the mss. msshist(i,j)
*  contains the mss for the jth scan on the ith iteration.
*  \param[out] phist Returns the history of the estimated parameters.
*  phist(i,j) contains the jth parameter on the ith iteration
*  \return A ptr to ReplacementManager that details which slices in
*  which scans were replaced by their expectations.
*
********************************************************************/
ReplacementManager *Register(// Input
                             const EddyCommandLineOptions&  clo,
                             ScanType                       st,
                             unsigned int                   niter,
                             const std::vector<float>&      fwhm,
                             SecondLevelECModel             slm,
                             bool                           dol,
                             // Input/Output
                             ECScanManager&                 sm,
                             ReplacementManager             *rm,
                             // Output
                             NEWMAT::Matrix&                msshist,
                             NEWMAT::Matrix&                phist) EddyTry
{
  static Utilities::FSLProfiler prof("_"+string(__FILE__)+"_"+string(__func__));
  double total_key = prof.StartEntry("Total");
  msshist.ReSize(niter,sm.NScans(st));
  phist.ReSize(niter,sm.NScans(st)*sm.Scan(0,st).NParam());
  double *mss_tmp = new double[sm.NScans(st)]; // Replace by vector
  if (rm == NULL) { // If no replacement manager passed in
    rm = new ReplacementManager(sm.NScans(st),static_cast<unsigned int>(sm.Scan(0,st).GetIma().zsize()),clo.OLDef(),clo.OLErrorType(),clo.OLType(),clo.MultiBand());
  }
  NEWIMAGE::volume<float> mask = sm.Scan(0,st).GetIma(); EddyUtils::SetTrilinearInterp(mask); mask = 1.0; // FOV-mask in model space

  for (unsigned int iter=0; iter<niter; iter++) {
    double load_key = prof.StartEntry("LoadPredictionMaker");
    // Load prediction maker in model space
    #ifdef COMPILE_GPU
    std::shared_ptr<DWIPredictionMaker> pmp = EddyGpuUtils::LoadPredictionMaker(clo,st,sm,iter,fwhm[iter],mask);
    #else
    std::shared_ptr<DWIPredictionMaker> pmp = EDDY::LoadPredictionMaker(clo,st,sm,iter,fwhm[iter],mask);
    #endif
    prof.EndEntry(load_key);
    // Detect outliers and replace them
    DiffStatsVector stats(sm.NScans(st));
    if (dol) {
      double dol_key = prof.StartEntry("Detecting and replacing outliers");
      std::shared_ptr<DWIPredictionMaker> od_pmp;
      #ifdef COMPILE_GPU
      od_pmp = pmp;
      stats = EddyGpuUtils::DetectOutliers(clo,st,od_pmp,mask,sm,*rm);
      if (iter) {
        EddyGpuUtils::ReplaceOutliers(clo,st,od_pmp,mask,*rm,false,sm);
        // EddyGpuUtils::UpdatePredictionMaker(clo,st,sm,rm,mask,pmp);
        pmp = EddyGpuUtils::LoadPredictionMaker(clo,st,sm,iter,fwhm[iter],mask);
      }
      #else
      od_pmp = pmp;
      stats = EDDY::DetectOutliers(clo,st,od_pmp,mask,sm,*rm);
      if (iter) {
        EDDY::ReplaceOutliers(clo,st,od_pmp,mask,*rm,false,sm);
        // EDDY::UpdatePredictionMaker(clo,st,sm,rm,mask,pmp);
        pmp = EDDY::LoadPredictionMaker(clo,st,sm,iter,fwhm[iter],mask);
      }
      #endif
      prof.EndEntry(dol_key);
    }
    //
    // Calculate the parameter updates
    // Note that this section will proceed in three different
    // ways depending on if it is run in single processor mode,
    // multi processor mode (OpenMP) or with a GPU-box (CUDA).
    //
    if (clo.Verbose()) cout << "Calculating parameter updates" << endl;
    double update_key = prof.StartEntry("Updating parameters");
    #ifdef COMPILE_GPU
    for (GpuPredictorChunk c(sm.NScans(st),mask); c<sm.NScans(st); c++) {
      double predict_chunk_key = prof.StartEntry("Predicting chunk");
      std::vector<unsigned int> si = c.Indicies();
      if (clo.VeryVerbose()) cout << "Making predictions for scans: " << c << endl;
      std::vector<NEWIMAGE::volume<float> > pred = pmp->Predict(si);
      prof.EndEntry(predict_chunk_key);
      if (clo.VeryVerbose()) cout << "Finished making predictions for scans: " << c << endl;
      double update_chunk_key = prof.StartEntry("Updating parameters for chunk");

      /*
      for (unsigned int i=0; i<si.size(); i++) {
        unsigned int global_indx = (st==EDDY::DWI) ? sm.GetDwi2GlobalIndexMapping(si[i]) : sm.Getb02GlobalIndexMapping(si[i]);
        if (clo.DebugLevel() && clo.DebugIndicies().IsAmongIndicies(global_indx)) {
          mss_tmp[si[i]] = EddyGpuUtils::MovAndECParamUpdate(pred[i],sm.GetSuscHzOffResField(si[i],st),sm.GetBiasField(),mask,true,fwhm[iter],global_indx,iter,clo.DebugLevel(),sm.Scan(si[i],st));
        }
        else mss_tmp[si[i]] = EddyGpuUtils::MovAndECParamUpdate(pred[i],sm.GetSuscHzOffResField(si[i],st),sm.GetBiasField(),mask,true,fwhm[iter],sm.Scan(si[i],st));
        if (clo.VeryVerbose()) printf("Iter: %d, scan: %d, gpu_mss = %f\n",iter,si[i],mss_tmp[si[i]]);
      }
      */
      tipl::par_for(si.size(),[&](unsigned int i) {
        unsigned int global_indx = (st==EDDY::DWI) ? sm.GetDwi2GlobalIndexMapping(si[i]) : sm.Getb02GlobalIndexMapping(si[i]);
        if (clo.DebugLevel() && clo.DebugIndicies().IsAmongIndicies(global_indx)) {
          mss_tmp[si[i]] = EddyGpuUtils::MovAndECParamUpdate(pred[i],sm.GetSuscHzOffResField(si[i],st),sm.GetBiasField(),mask,true,fwhm[iter],global_indx,iter,clo.DebugLevel(),sm.Scan(si[i],st));
        }
        else mss_tmp[si[i]] = EddyGpuUtils::MovAndECParamUpdate(pred[i],sm.GetSuscHzOffResField(si[i],st),sm.GetBiasField(),mask,true,fwhm[iter],sm.Scan(si[i],st));
        if (clo.VeryVerbose()) printf("Iter: %d, scan: %d, gpu_mss = %f\n",iter,si[i],mss_tmp[si[i]]);
      });
      prof.EndEntry(update_chunk_key);
    }
    #else
    # pragma omp parallel for shared(mss_tmp, pmp)
    tipl::par_for(int(sm.NScans(st)),[&](int s){
      // Get prediction in model space
      NEWIMAGE::volume<float> pred = pmp->Predict(s);
      // Update parameters
      unsigned int global_indx = (st==EDDY::DWI) ? sm.GetDwi2GlobalIndexMapping(s) : sm.Getb02GlobalIndexMapping(s);
      if (clo.DebugLevel() && clo.DebugIndicies().IsAmongIndicies(global_indx)) {
        mss_tmp[s] = EddyUtils::MovAndECParamUpdate(pred,sm.GetSuscHzOffResField(s,st),sm.GetBiasField(),mask,true,fwhm[iter],global_indx,iter,clo.DebugLevel(),sm.Scan(s,st));
      }
      else mss_tmp[s] = EddyUtils::MovAndECParamUpdate(pred,sm.GetSuscHzOffResField(s,st),sm.GetBiasField(),mask,true,fwhm[iter],sm.Scan(s,st));
      if (clo.VeryVerbose()) printf("Iter: %d, scan: %d, mss = %f\n",iter,s,mss_tmp[s]);
    }
    );
    #endif
    prof.EndEntry(update_key);

    // Print/collect some information that can be used for diagnostics
    Diagnostics(clo,iter,st,sm,mss_tmp,stats,*rm,msshist,phist);

    // Maybe use model based EC parameters
    if (slm != No_2nd_lvl_mdl) {
      if (clo.VeryVerbose()) cout << "Performing 2nd level modelling of estimated parameters" << endl;
      sm.SetPredictedECParam(st,slm);
    }

    // Maybe try and separate field-offset and translation in PE direction
    if (clo.SeparateOffsetFromMovement()) {
      if (clo.VeryVerbose()) cout << "Attempting to separate field-offset from subject movement" << endl;
      sm.SeparateFieldOffsetFromMovement(st,clo.OffsetModel());
    }
  }

  delete [] mss_tmp;
  prof.EndEntry(total_key);
  return(rm);
} EddyCatch

ReplacementManager *FinalOLCheck(// Input
                                 const EddyCommandLineOptions&  clo,
                                 // Input/output
                                 ReplacementManager             *rm,
                                 ECScanManager&                 sm) EddyTry
{
  static Utilities::FSLProfiler prof("_"+string(__FILE__)+"_"+string(__func__));
  double total_key = prof.StartEntry("Total");
  NEWIMAGE::volume<float> mask = sm.Scan(0,DWI).GetIma(); EddyUtils::SetTrilinearInterp(mask); mask = 1.0; // FOV-mask in model space
  if (rm == NULL) {
    rm = new ReplacementManager(sm.NScans(DWI),static_cast<unsigned int>(sm.Scan(0,DWI).GetIma().zsize()),clo.OLDef(),clo.OLErrorType(),clo.OLType(),clo.MultiBand());
  }

  // Load prediction maker in model space
  double load_key = prof.StartEntry("LoadPredictionMaker");
  #ifdef COMPILE_GPU
  std::shared_ptr<DWIPredictionMaker> pmp = EddyGpuUtils::LoadPredictionMaker(clo,DWI,sm,0,0.0,mask);
  #else
  std::shared_ptr<DWIPredictionMaker> pmp = EDDY::LoadPredictionMaker(clo,DWI,sm,0,0.0,mask);
  #endif
  prof.EndEntry(load_key);
  // Detect outliers and replace them
  DiffStatsVector stats(sm.NScans(DWI));
  bool add_noise = clo.AddNoiseToReplacements();
  double det_key = prof.StartEntry("DetectOutliers");
  #ifdef COMPILE_GPU
  stats = EddyGpuUtils::DetectOutliers(clo,DWI,pmp,mask,sm,*rm);
  #else
  stats = EDDY::DetectOutliers(clo,DWI,pmp,mask,sm,*rm);
  #endif
  prof.EndEntry(det_key);
  double rep_key = prof.StartEntry("ReplaceOutliers");
  #ifdef COMPILE_GPU
  EddyGpuUtils::ReplaceOutliers(clo,DWI,pmp,mask,*rm,add_noise,sm);
  #else
  EDDY::ReplaceOutliers(clo,DWI,pmp,mask,*rm,add_noise,sm);
  #endif
  prof.EndEntry(rep_key);
  prof.EndEntry(total_key);
  return(rm);
} EddyCatch

/****************************************************************//**
*
*  A global function that loads up a prediction maker with all scans
*  of a given type. It will load it with unwarped scans (given the
*  current estimates of the warps) as served up by sm.GetUnwarpedScan()
*  or sm.GetUnwarpedOrigScan() depending on the value of use_orig.
*  \param[in] clo Carries information about the command line options
*  that eddy was invoked with.
*  \param[in] st Specifies if we should register the diffusion weighted
*  images or the b=0 images. If it is set to DWI the function will return
*  an EDDY::DiffusionGP prediction maker and if it is set to b0 it will
*  return an EDDY::b0Predictor.
*  \param[in] sm Collection of all scans.
*  \param[out] mask Returns a mask that indicates the voxels where data
*  is present for all input scans in sm.
*  \param[in] use_orig If set to true it will load it with unwarped "original"
*  , i.e. un-smoothed, scans. Default is false.
*  \return A safe pointer to a DWIPredictionMaker that can be used to
*  make predictions about what the scans should look like in undistorted space.
*
********************************************************************/
std::shared_ptr<DWIPredictionMaker> LoadPredictionMaker(// Input
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
  static Utilities::FSLProfiler prof("_"+string(__FILE__)+"_"+string(__func__));
  double total_key = prof.StartEntry("Total");
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
  else pmp = std::shared_ptr<DWIPredictionMaker>(new b0Predictor);  // Silly mean predictor for b=0 data
  pmp->SetNoOfScans(sm.NScans(st));
  mask = sm.Scan(0,st).GetIma(); EddyUtils::SetTrilinearInterp(mask); mask = 1.0;

  double load_key = prof.StartEntry("Loading");
  if (clo.Verbose()) cout << "Loading prediction maker";
  if (clo.VeryVerbose()) cout << endl << "Scan: " << endl;
#pragma omp parallel for shared(pmp,st)
  std::mutex mu;
  tipl::par_for(int(sm.NScans(st)),[&](int s){
    if (clo.VeryVerbose()) printf(" %d\n",s);
    NEWIMAGE::volume<float> tmpmask = sm.Scan(s,st).GetIma();
    EddyUtils::SetTrilinearInterp(tmpmask); tmpmask = 1.0;
    if (use_orig) pmp->SetScan(sm.GetUnwarpedOrigScan(s,tmpmask,st),sm.Scan(s,st).GetDiffPara(clo.RotateBVecsDuringEstimation()),s);
    else pmp->SetScan(sm.GetUnwarpedScan(s,tmpmask,st),sm.Scan(s,st).GetDiffPara(clo.RotateBVecsDuringEstimation()),s);
#pragma omp critical
    {
      std::lock_guard<std::mutex> lock(mu);
      mask *= tmpmask;
    }
  }
  );
  prof.EndEntry(load_key);
  double eval_key = prof.StartEntry("Evaluating");
  if (clo.Verbose()) cout << endl << "Evaluating prediction maker model" << endl;
  pmp->EvaluateModel(sm.Mask()*mask,fwhm,clo.Verbose());
  prof.EndEntry(eval_key);
  if (clo.DebugLevel() > 2 && st==DWI) {
    char fname[256];
    sprintf(fname,"EDDY_DEBUG_K_Mat_Data_%02d",iter);
    pmp->WriteMetaData(fname);
  }

  prof.EndEntry(total_key);
  return(pmp);
} EddyCatch

DiffStatsVector DetectOutliers(// Input
                               const EddyCommandLineOptions&             clo,
                               ScanType                                  st,
                               const std::shared_ptr<DWIPredictionMaker> pmp,
                               const NEWIMAGE::volume<float>&            mask,
                               const ECScanManager&                      sm,
                               // Input/Output
                               ReplacementManager&                       rm) EddyTry
{
  static Utilities::FSLProfiler prof("_"+string(__FILE__)+"_"+string(__func__));
  double total_key = prof.StartEntry("Total");
  if (clo.VeryVerbose()) cout << "Checking for outliers" << endl;
  // Generate slice-wise stats on difference between observation and prediction
  DiffStatsVector stats(sm.NScans(st));
#pragma omp parallel for shared(stats,st)
  tipl::par_for(int(sm.NScans(st)),[&](int s){
    if (clo.VeryVerbose()) cout << s << std::endl;
    NEWIMAGE::volume<float> pred = pmp->Predict(s);
    stats[s] = EddyUtils::GetSliceWiseStats(pred,sm.GetSuscHzOffResField(s,st),mask,sm.Mask(),sm.Scan(s,st));
  }
  );
  // Detect outliers and update replacement manager
  rm.Update(stats);
  prof.EndEntry(total_key);
  return(stats);
} EddyCatch

void ReplaceOutliers(// Input
                     const EddyCommandLineOptions&             clo,
                     ScanType                                  st,
                     const std::shared_ptr<DWIPredictionMaker> pmp,
                     const NEWIMAGE::volume<float>&            mask,
                     const ReplacementManager&                 rm,
                     bool                                      add_noise,
                     // Input/Output
                     ECScanManager&                            sm) EddyTry
{
  static Utilities::FSLProfiler prof("_"+string(__FILE__)+"_"+string(__func__));
  double total_key = prof.StartEntry("Total");
  // Replace outlier slices with their predictions
  if (clo.VeryVerbose()) cout << "Replacing outliers with predictions" << endl;
#pragma omp parallel for shared(st)
  tipl::par_for(int(sm.NScans(st)),[&](int s){
    std::vector<unsigned int> ol = rm.OutliersInScan(s);
    if (ol.size()) { // If this scan has outlier slices
      if (clo.VeryVerbose()) cout << "Scan " << s << " has " << ol.size() << " outlier slices" << endl;
      NEWIMAGE::volume<float> pred = pmp->Predict(s,true);
      if (add_noise) {
        double vp = pmp->PredictionVariance(s,true);
        double ve = pmp->ErrorVariance(s);
        double stdev = std::sqrt(vp+ve) - std::sqrt(vp);
        pred += EddyUtils::MakeNoiseIma(pred,0.0,stdev);
      }
      sm.Scan(s,st).SetAsOutliers(pred,sm.GetSuscHzOffResField(s,st),mask,ol);
    }
  }
  );
  prof.EndEntry(total_key);
  return;
} EddyCatch

/****************************************************************//**
*
*  A global function that makes ff=1 predictions, in model space,
*  for all scans of type st.
*  \param[in] clo Carries information about the command line options
*  that eddy was invoked with.
*  \param[in] st Specifies if we should resample the diffusion weighted
*  images, the b=0 images or both.
*  \param[in] sm Collection of all scans.
*  \param[out] pred A 4D volume with predictions for all scans of the
*  type indicated by st.
*  \return The hyper-parameters used for the predictions
*
********************************************************************/
std::vector<double> GetPredictionsForResampling(// Input
                                                const EddyCommandLineOptions&    clo,
                                                ScanType                         st,
                                                const ECScanManager&             sm,
                                                // Output
                                                NEWIMAGE::volume4D<float>&       pred) EddyTry
{
  static Utilities::FSLProfiler prof("_"+string(__FILE__)+"_"+string(__func__));
  double total_key = prof.StartEntry("Total");
  pred.reinitialize(sm.Scan(0,st).GetIma().xsize(),sm.Scan(0,st).GetIma().ysize(),sm.Scan(0,st).GetIma().zsize(),sm.NScans(st));
  NEWIMAGE::copybasicproperties(sm.Scan(0,st).GetIma(),pred);
  NEWIMAGE::volume<float> mask = sm.Scan(0,st).GetIma(); EddyUtils::SetTrilinearInterp(mask); mask = 1.0; // FOV-mask in model space
  EddyCommandLineOptions lclo = clo;
  std::vector<double> hypar;
  if (lclo.HyParFudgeFactor() != 1.0) lclo.SetHyParFudgeFactor(1.0);
  if (st == ANY || st == B0) {
    #ifdef COMPILE_GPU
    std::shared_ptr<DWIPredictionMaker> pmp = EddyGpuUtils::LoadPredictionMaker(lclo,B0,sm,0,0.0,mask);
    #else
    std::shared_ptr<DWIPredictionMaker> pmp = EDDY::LoadPredictionMaker(lclo,B0,sm,0,0.0,mask);
    #endif
    for (unsigned int s=0; s<sm.NScans(B0); s++) {
      if (st == B0) pred[s] = pmp->Predict(s,true);
      else pred[sm.Getb02GlobalIndexMapping(s)] = pmp->Predict(s,true);
    }
  }
  if (st == ANY || st == DWI) {
    #ifdef COMPILE_GPU
    std::shared_ptr<DWIPredictionMaker> pmp = EddyGpuUtils::LoadPredictionMaker(lclo,DWI,sm,0,0.0,mask);
    #else
    std::shared_ptr<DWIPredictionMaker> pmp = EDDY::LoadPredictionMaker(lclo,DWI,sm,0,0.0,mask);
    #endif
    hypar = pmp->GetHyperPar();
    for (unsigned int s=0; s<sm.NScans(DWI); s++) {
      if (st == DWI) pred[s] = pmp->Predict(s,true);
      else pred[sm.GetDwi2GlobalIndexMapping(s)] = pmp->Predict(s,true);
    }
  }
  prof.EndEntry(total_key);
  return(hypar);
} EddyCatch

/****************************************************************//**
*
*  A global function that makes ff=1 predictions, in model space,
*  for all scans of type st. It differs from the "normal" way eddy
*  makes predictions in that it uses a "scattered data reconstruction"
*  approach.
*  \param[in] clo Carries information about the command line options
*  that eddy was invoked with.
*  \param[in] st Specifies if we should resample the diffusion weighted
*  images, the b=0 images or both.
*  \param[in] sm Collection of all scans.
*  \param[in] hypar Hyperparameters. Valid only for "spherical" covariance-
*  function.
*  \param[out] pred A 4D volume with predictions for all scans of the
*  type indicated by st.
*  \param[in] vwbvrot If true means that bvec rotation is per volume
*  instead of per slice/MB-group which is default.
*
********************************************************************/
void GetScatterBrainPredictions(// Input
                                const EddyCommandLineOptions&    clo,
                                ScanType                         st,
                                ECScanManager&                   sm,
                                const std::vector<double>&       hypar,
                                // Output
                                NEWIMAGE::volume4D<float>&       pred,
                                // Optional input
                                bool                             vwbvrot) EddyTry
{
  static Utilities::FSLProfiler prof("_"+string(__FILE__)+"_"+string(__func__));
  double total_key = prof.StartEntry("Total");
  // Do some checking to ensure that the call and the input makes sense
  if (clo.CovarianceFunction() != EDDY::NewSpherical) throw EddyException("EDDY::GetScatterBrainPredictions: Predictions only available for Spherical covariance function");
  if (!clo.IsSliceToVol()) throw EddyException("EDDY::GetScatterBrainPredictions: Predictions only makes sense for slice-to-vol model");
  NEWIMAGE::volume<float> mask = sm.Scan(0,st).GetIma(); EddyUtils::SetTrilinearInterp(mask); mask = 1.0; // FOV-mask in model space
  std::vector<DiffPara> dp = sm.GetDiffParas(DWI);
  NewSphericalKMatrix K(clo.DontCheckShelling());
  K.SetDiffusionPar(dp);
  if (hypar.size() != K.NoOfHyperPar()) throw EddyException("EDDY::GetScatterBrainPredictions: Incompatible hypar size");
  pred.reinitialize(sm.Scan(0,st).GetIma().xsize(),sm.Scan(0,st).GetIma().ysize(),sm.Scan(0,st).GetIma().zsize(),sm.NScans(st));
  NEWIMAGE::copybasicproperties(sm.Scan(0,st).GetIma(),pred);
  // The B0s predictions are just the means and unaffected by rotations
  // Should be updated to be closer to the way we calculate the predictions
  // for the diffusion weighted data.
  if (st==B0 || st==ANY) {
    EDDY::PolationPara old_pol = sm.GetPolation();
    EDDY::PolationPara new_pol = old_pol;
    if (clo.IsSliceToVol() && new_pol.GetS2VInterp() != NEWIMAGE::trilinear) new_pol.SetS2VInterp(NEWIMAGE::trilinear);
    sm.SetPolation(new_pol);
    #ifdef COMPILE_GPU
    std::shared_ptr<DWIPredictionMaker> pmp = EddyGpuUtils::LoadPredictionMaker(clo,B0,sm,0,0.0,mask);
    #else
    std::shared_ptr<DWIPredictionMaker> pmp = EDDY::LoadPredictionMaker(clo,B0,sm,0,0.0,mask);
    #endif
    for (unsigned int s=0; s<sm.NScans(B0); s++) {
      if (st == B0) pred[s] = pmp->Predict(s,true);
      else pred[sm.Getb02GlobalIndexMapping(s)] = pmp->Predict(s,true);
    }
    sm.SetPolation(new_pol);
  }
  if (st==DWI || st==ANY) {
    #ifdef COMPILE_GPU
    EddyGpuUtils::MakeScatterBrainPredictions(clo,sm,hypar,pred,vwbvrot);
    #else
    throw EddyException("EDDY::GetScatterBrainPredictions: Only implemented for GPU");
    #endif
  }
  prof.EndEntry(total_key);
  return;
} EddyCatch

/****************************************************************//**
*
*  A global function that calculates CNR-maps for the dwi volume,
*  SNR-maps for the b0 volumes and a 4D map of residuals.
*  for all scans of type st. All the output parameters are optional
*  and passing in a nullptr means that parameter is not calculated
*  or returned.
*  \param[in] clo Carries information about the command line options
*  that eddy was invoked with.
*  \param[in] sm Collection of all scans.
*  \param[out] std_cnr A 4D file with one CNR-map, calculated as
*  std(pred)/std(res), per shell.
*  \param[out] range_cnr A 4D file with one CNR-map,calculated as
*  range(pred)/std(res), per shell.
*  \param[out] b0_snr A 3D SNR-map for the b0-volumes
*  \param[out] residuals A 4D-map with the residuals for both dwis and b0s
*
********************************************************************/
void CalculateCNRMaps(// Input
                      const EddyCommandLineOptions&               clo,
                      const ECScanManager&                        sm,
                      // Output
                      std::shared_ptr<NEWIMAGE::volume4D<float> > std_cnr,
                      std::shared_ptr<NEWIMAGE::volume4D<float> > range_cnr,
                      std::shared_ptr<NEWIMAGE::volume<float> >   b0_snr,
                      std::shared_ptr<NEWIMAGE::volume4D<float> > residuals) EddyTry
{
  static Utilities::FSLProfiler prof("_"+string(__FILE__)+"_"+string(__func__));
  double total_key = prof.StartEntry("Total");
  if (std_cnr==nullptr && range_cnr==nullptr && b0_snr==nullptr && residuals==nullptr) {
    throw EddyException("EDDY::CalculateCNRMaps: At least one output parameter must be set.");
  }
  if (!sm.IsShelled()) {
    throw EddyException("EDDY::CalculateCNRMaps: Can only calculate CNR for shelled data.");
  }
  if ((std_cnr!=nullptr && (!NEWIMAGE::samesize(sm.Scan(0,ANY).GetIma(),(*std_cnr)[0]) || (*std_cnr).tsize() != sm.NoOfShells(DWI))) ||
      (range_cnr!=nullptr && (!NEWIMAGE::samesize(sm.Scan(0,ANY).GetIma(),(*range_cnr)[0]) || (*range_cnr).tsize() != sm.NoOfShells(DWI))) ||
      (b0_snr!=nullptr && !NEWIMAGE::samesize(sm.Scan(0,ANY).GetIma(),*b0_snr)) ||
      (residuals!=nullptr && (!NEWIMAGE::samesize(sm.Scan(0,ANY).GetIma(),(*residuals)[0]) || (*residuals).tsize() != sm.NScans(ANY)))) {
    throw EddyException("EDDY::CalculateCNRMaps: Size mismatch between sm and output containers.");
  }
  NEWIMAGE::volume<float> mask = sm.Scan(0,DWI).GetIma(); // FOV-mask in model space
  EddyUtils::SetTrilinearInterp(mask); mask = 1.0;
  std::shared_ptr<DWIPredictionMaker> dwi_pmp;
  std::shared_ptr<DWIPredictionMaker> b0_pmp;
  if (std_cnr || range_cnr || residuals) {
    #ifdef COMPILE_GPU
    dwi_pmp = EddyGpuUtils::LoadPredictionMaker(clo,DWI,sm,0,0.0,mask);
    #else
    dwi_pmp = EDDY::LoadPredictionMaker(clo,DWI,sm,0,0.0,mask);
    #endif
  }
  if (b0_snr || residuals) {
    #ifdef COMPILE_GPU
    b0_pmp = EddyGpuUtils::LoadPredictionMaker(clo,B0,sm,0,0.0,mask);
    #else
    b0_pmp = EDDY::LoadPredictionMaker(clo,B0,sm,0,0.0,mask);
    #endif
  }
  if (std_cnr || range_cnr) { // Calculate and write shell-wise CNR maps
    std::vector<double>                      grpb;                                 // b-values of the different dwi shells
    std::vector<std::vector<unsigned int> >  dwi_indx = sm.GetShellIndicies(grpb); // Global indicies of dwi scans
    std::vector<NEWIMAGE::volume<float> >    mvols(grpb.size());                   // Volumes of mean predictions for the shells
    // Calculate mean volumes of predictions
    for (unsigned int i=0; i<grpb.size(); i++) {
      mvols[i] = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][0]));
      for (unsigned int j=1; j<dwi_indx[i].size(); j++) {
        mvols[i] += dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j]));
      }
      mvols[i] /= float(dwi_indx[i].size());
    }
    std::vector<NEWIMAGE::volume<float> >    stdvols(grpb.size());
    if (std_cnr) { // Calculate standard deviation volumes of predictions
      for (unsigned int i=0; i<grpb.size(); i++) {
        NEWIMAGE::volume<float> tmp = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][0])) - mvols[i];
        stdvols[i] = tmp*tmp;
        for (unsigned int j=1; j<dwi_indx[i].size(); j++) {
          tmp = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j])) - mvols[i];
          stdvols[i] += tmp*tmp;
        }
        stdvols[i] /= float(dwi_indx[i].size()-1);
        stdvols[i] = NEWIMAGE::sqrt(stdvols[i]);
      }
    }
    std::vector<NEWIMAGE::volume<float> >    minvols(grpb.size());
    std::vector<NEWIMAGE::volume<float> >    maxvols(grpb.size());
    if (range_cnr) { // Caclculate range (max-min) volumes of predictions
      for (unsigned int i=0; i<grpb.size(); i++) {
        minvols[i] = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][0]));
        maxvols[i] = minvols[i];
        for (unsigned int j=1; j<dwi_indx[i].size(); j++) {
          minvols[i] = NEWIMAGE::min(minvols[i],dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j])));
          maxvols[i] = NEWIMAGE::max(maxvols[i],dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j])));
        }
        maxvols[i] -= minvols[i]; // Maxvols now contain the range rather than the max
      }
    }
    // Calculate standard deviation of residuals
    std::vector<NEWIMAGE::volume<float> >  stdres(grpb.size());
    for (unsigned int i=0; i<grpb.size(); i++) {
      NEWIMAGE::volume<float> tmp = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][0])) - dwi_pmp->InputData(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][0]));
      stdres[i] = tmp*tmp;
      for (unsigned int j=1; j<dwi_indx[i].size(); j++) {
        tmp = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j])) - dwi_pmp->InputData(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j]));
        stdres[i] += tmp*tmp;
      }
      stdres[i] /= float(dwi_indx[i].size()-1);
      stdres[i] = NEWIMAGE::sqrt(stdres[i]);
    }
    // Divide prediction std/range with std of residuals to get CNR
    for (unsigned int i=0; i<grpb.size(); i++) {
      if (std_cnr) (*std_cnr)[i] = stdvols[i] / stdres[i];
      if (range_cnr) (*range_cnr)[i] = maxvols[i] / stdres[i];
    }
  }
  // Get the SNR of the b0s
  if (b0_snr) {
    if (sm.NScans(B0) > 1) {
      std::vector<unsigned int>   b0_indx = sm.GetB0Indicies();
      *b0_snr = b0_pmp->Predict(0) - b0_pmp->InputData(0); // N.B. Predict(i) is mean of all b0 scans
      *b0_snr *= *b0_snr;
      for (unsigned int i=1; i<b0_indx.size(); i++) {
        NEWIMAGE::volume<float> tmp = b0_pmp->Predict(i) - b0_pmp->InputData(i);
        *b0_snr += tmp*tmp;
      }
      *b0_snr /= static_cast<float>(b0_indx.size()-1);
      *b0_snr = NEWIMAGE::sqrt(*b0_snr);
      *b0_snr = b0_pmp->Predict(0) / *b0_snr;
    }
    else (*b0_snr) = 0.0; // Set SNR to zero if we can't estimate it
  }
  // Get residuals
  if (residuals) {
    for (unsigned int i=0; i<sm.NScans(DWI); i++) {
      (*residuals)[sm.GetDwi2GlobalIndexMapping(i)] = dwi_pmp->InputData(i) - dwi_pmp->Predict(i);
    }
    for (unsigned int i=0; i<sm.NScans(B0); i++) {
      (*residuals)[sm.Getb02GlobalIndexMapping(i)] = b0_pmp->InputData(i) - b0_pmp->Predict(i);
    }
  }
  prof.EndEntry(total_key);

  return;
} EddyCatch

void WriteCNRMaps(// Input
                  const EddyCommandLineOptions&   clo,
                  const ECScanManager&            sm,
                  const std::string&              spatial_fname,
                  const std::string&              range_fname,
                  const std::string&              residual_fname) EddyTry
{
  static Utilities::FSLProfiler prof("_"+string(__FILE__)+"_"+string(__func__));
  double total_key = prof.StartEntry("Total");
  if (spatial_fname.empty() && residual_fname.empty()) throw EddyException("EDDY::WriteCNRMaps: At least one of spatial and residual fname must be set");

  // Allocate memory for the maps we are interested in
  std::shared_ptr<NEWIMAGE::volume4D<float> > std_cnr;
  std::shared_ptr<NEWIMAGE::volume4D<float> > range_cnr;
  std::shared_ptr<NEWIMAGE::volume<float> >   b0_snr;
  std::shared_ptr<NEWIMAGE::volume4D<float> > residuals;
  if (!spatial_fname.empty()) {
    const NEWIMAGE::volume<float>& tmp = sm.Scan(0,ANY).GetIma();
    std_cnr = std::make_shared<NEWIMAGE::volume4D<float> > (tmp.xsize(),tmp.ysize(),tmp.zsize(),sm.NoOfShells(DWI));
    NEWIMAGE::copybasicproperties(tmp,*std_cnr);
    b0_snr = std::make_shared<NEWIMAGE::volume<float> > (tmp.xsize(),tmp.ysize(),tmp.zsize());
    NEWIMAGE::copybasicproperties(tmp,*b0_snr);
  }
  if (!range_fname.empty()) {
    const NEWIMAGE::volume<float>& tmp = sm.Scan(0,ANY).GetIma();
    range_cnr = std::make_shared<NEWIMAGE::volume4D<float> > (tmp.xsize(),tmp.ysize(),tmp.zsize(),sm.NoOfShells(DWI));
    NEWIMAGE::copybasicproperties(tmp,*range_cnr);
    if (b0_snr==nullptr) {
      b0_snr = std::make_shared<NEWIMAGE::volume<float> > (tmp.xsize(),tmp.ysize(),tmp.zsize());
      NEWIMAGE::copybasicproperties(tmp,*b0_snr);
    }
  }
  if (!residual_fname.empty()) {
    const NEWIMAGE::volume<float>& tmp = sm.Scan(0,ANY).GetIma();
    residuals = std::make_shared<NEWIMAGE::volume4D<float> > (tmp.xsize(),tmp.ysize(),tmp.zsize(),sm.NScans(ANY));
    NEWIMAGE::copybasicproperties(tmp,*residuals);
  }
  // Calculate the maps we are interested in
  EDDY::CalculateCNRMaps(clo,sm,std_cnr,range_cnr,b0_snr,residuals);
  // Write them out
  if (!spatial_fname.empty()) {
    const NEWIMAGE::volume<float>& tmp = sm.Scan(0,ANY).GetIma();
    NEWIMAGE::volume4D<float> ovol(tmp.xsize(),tmp.ysize(),tmp.zsize(),sm.NoOfShells(ANY));
    NEWIMAGE::copybasicproperties(tmp,ovol);
    ovol[0] = *b0_snr;
    for (unsigned int i=0; i<sm.NoOfShells(DWI); i++) ovol[i+1] = (*std_cnr)[i];
    NEWIMAGE::write_volume(ovol,spatial_fname);
  }
  if (!range_fname.empty()) {
    const NEWIMAGE::volume<float>& tmp = sm.Scan(0,ANY).GetIma();
    NEWIMAGE::volume4D<float> ovol(tmp.xsize(),tmp.ysize(),tmp.zsize(),sm.NoOfShells(ANY));
    NEWIMAGE::copybasicproperties(tmp,ovol);
    ovol[0] = *b0_snr;
    for (unsigned int i=0; i<sm.NoOfShells(DWI); i++) ovol[i+1] = (*range_cnr)[i];
    NEWIMAGE::write_volume(ovol,spatial_fname);
  }
  if (!residual_fname.empty()) NEWIMAGE::write_volume(*residuals,residual_fname);
  prof.EndEntry(total_key);

  return;
} EddyCatch

/*
void WriteCNRMaps(// Input
                  const EddyCommandLineOptions&   clo,
                  const ECScanManager&            sm,
                  const std::string&              spatial_fname,
                  const std::string&              range_fname,
                  const std::string&              residual_fname) EddyTry
{
  if (spatial_fname == std::string("") && residual_fname == std::string("")) throw EddyException("EDDY::WriteCNRMaps: At least one of spatial and residual fname must be set");

  // Load prediction maker
  NEWIMAGE::volume<float> mask = sm.Scan(0,DWI).GetIma(); EddyUtils::SetTrilinearInterp(mask); mask = 1.0; // FOV-mask in model space
  #ifdef COMPILE_GPU
  std::shared_ptr<DWIPredictionMaker> dwi_pmp = EddyGpuUtils::LoadPredictionMaker(clo,DWI,sm,0,0.0,mask);
  std::shared_ptr<DWIPredictionMaker> b0_pmp = EddyGpuUtils::LoadPredictionMaker(clo,B0,sm,0,0.0,mask);
  #else
  std::shared_ptr<DWIPredictionMaker> dwi_pmp = EDDY::LoadPredictionMaker(clo,DWI,sm,0,0.0,mask);
  std::shared_ptr<DWIPredictionMaker> b0_pmp = EDDY::LoadPredictionMaker(clo,B0,sm,0,0.0,mask);
  #endif

  if (sm.IsShelled()) {
    if (spatial_fname != std::string("") || range_fname != std::string("")) {  // Calculate and write shell-wise CNR maps
      std::vector<double>                      grpb;
      std::vector<std::vector<unsigned int> >  dwi_indx = sm.GetShellIndicies(grpb); // Global indicies of dwi scans
      std::vector<NEWIMAGE::volume<float> >    mvols(grpb.size());
      // Calculate mean volumes of predictions
      for (unsigned int i=0; i<grpb.size(); i++) {
        mvols[i] = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][0]));
        for (unsigned int j=1; j<dwi_indx[i].size(); j++) {
          mvols[i] += dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j]));
        }
        mvols[i] /= float(dwi_indx[i].size());
      }
      // Calculate standard deviation volumes of predictions
      std::vector<NEWIMAGE::volume<float> >    stdvols(grpb.size());
      for (unsigned int i=0; i<grpb.size(); i++) {
        NEWIMAGE::volume<float> tmp = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][0])) - mvols[i];
        stdvols[i] = tmp*tmp;
        for (unsigned int j=1; j<dwi_indx[i].size(); j++) {
          tmp = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j])) - mvols[i];
          stdvols[i] += tmp*tmp;
        }
        stdvols[i] /= float(dwi_indx[i].size()-1);
        stdvols[i] = NEWIMAGE::sqrt(stdvols[i]);
      }
      // Calculate range (min--max) volumes of predictions to make dHCP data seem like it has decent CNR
      std::vector<NEWIMAGE::volume<float> >    minvols(grpb.size());
      std::vector<NEWIMAGE::volume<float> >    maxvols(grpb.size());
      for (unsigned int i=0; i<grpb.size(); i++) {
        minvols[i] = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][0]));
        maxvols[i] = minvols[i];
        for (unsigned int j=1; j<dwi_indx[i].size(); j++) {
          minvols[i] = NEWIMAGE::min(minvols[i],dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j])));
          maxvols[i] = NEWIMAGE::max(maxvols[i],dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j])));
        }
        maxvols[i] -= minvols[i]; // Maxvols now contain the range rather than the max
      }
      // Calculate standard deviation of residuals
      std::vector<NEWIMAGE::volume<float> >  stdres(grpb.size());
      for (unsigned int i=0; i<grpb.size(); i++) {
        NEWIMAGE::volume<float> tmp = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][0])) - dwi_pmp->InputData(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][0]));
        stdres[i] = tmp*tmp;
        for (unsigned int j=1; j<dwi_indx[i].size(); j++) {
          tmp = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j])) - dwi_pmp->InputData(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j]));
          stdres[i] += tmp*tmp;
        }
        stdres[i] /= float(dwi_indx[i].size()-1);
        stdres[i] = NEWIMAGE::sqrt(stdres[i]);
      }
      // Divide std and range of predictions with std of residuals to get CNR
      for (unsigned int i=0; i<grpb.size(); i++) {
        maxvols[i] /= stdres[i];
        stdvols[i] /= stdres[i];
      }
      // Get the SNR (since CNR is zero) of the b0s
      std::vector<unsigned int>   b0_indx = sm.GetB0Indicies();
      NEWIMAGE::volume<float>     b0_SNR = b0_pmp->Predict(0) - b0_pmp->InputData(0);
      if (b0_indx.size() > 1) {
        b0_SNR *= b0_SNR;
        for (unsigned int i=1; i<b0_indx.size(); i++) {
          NEWIMAGE::volume<float> tmp = b0_pmp->Predict(i) - b0_pmp->InputData(i);
          b0_SNR += tmp*tmp;
        }
        b0_SNR /= float(b0_indx.size()-1);
        b0_SNR = NEWIMAGE::sqrt(b0_SNR);
        b0_SNR = b0_pmp->Predict(0) /= b0_SNR;
      }
      else b0_SNR = 0.0; // Set it to zero if we can't assess it.
      // Put SNR and CNR maps together into 4D file with spatial CNR
      NEWIMAGE::volume4D<float> spat_cnr(stdvols[0].xsize(),stdvols[0].ysize(),stdvols[0].zsize(),stdvols.size()+1);
      NEWIMAGE::copybasicproperties(stdvols[0],spat_cnr);
      spat_cnr[0] = b0_SNR;
      if (spatial_fname != std::string("")) {
        for (unsigned int i=0; i<stdvols.size(); i++) spat_cnr[i+1] = stdvols[i];
        NEWIMAGE::write_volume(spat_cnr,spatial_fname);
      }
      // Put SNR and range maps together into 4D file with spatial "range-CNR"
      if (range_fname != std::string("")) {
        for (unsigned int i=0; i<maxvols.size(); i++) spat_cnr[i+1] = maxvols[i];
        NEWIMAGE::write_volume(spat_cnr,range_fname);
      }
    }
    if (residual_fname != std::string("")) {   // Calculate and write maps of residuals for all (b0 and DWI) scans
      NEWIMAGE::volume4D<float> residuals(dwi_pmp->InputData(0).xsize(),dwi_pmp->InputData(0).ysize(),dwi_pmp->InputData(0).zsize(),sm.NScans(ANY));
      NEWIMAGE::copybasicproperties(dwi_pmp->InputData(0),residuals);
      for (unsigned int i=0; i<sm.NScans(DWI); i++) {
        residuals[sm.GetDwi2GlobalIndexMapping(i)] = dwi_pmp->InputData(i) - dwi_pmp->Predict(i);
      }
      for (unsigned int i=0; i<sm.NScans(B0); i++) {
        residuals[sm.Getb02GlobalIndexMapping(i)] = b0_pmp->InputData(i) - b0_pmp->Predict(i);
      }
      NEWIMAGE::write_volume(residuals,residual_fname);
    }
  }
  else {
    throw EddyException("WriteCNRMaps: Cannot calculate CNR for non-shelled data.");
  }
} EddyCatch
*/

void Diagnostics(// Input
                 const EddyCommandLineOptions&  clo,
                 unsigned int                   iter,
                 ScanType                       st,
                 const ECScanManager&           sm,
                 const double                   *mss_tmp,
                 const DiffStatsVector&         stats,
                 const ReplacementManager&      rm,
                 // Output
                 NEWMAT::Matrix&                mss,
                 NEWMAT::Matrix&                phist) EddyTry
{
  if (clo.Verbose()) {
    double tss=0.0;
    for (unsigned int s=0; s<sm.NScans(st); s++) tss+=mss_tmp[s];
    cout << "Iter: " << iter << ", Total mss = " << tss/sm.NScans(st) << endl;
  }

  for (unsigned int s=0; s<sm.NScans(st); s++) {
    mss(iter+1,s+1) = mss_tmp[s];
    phist.SubMatrix(iter+1,iter+1,s*sm.Scan(0,st).NParam()+1,(s+1)*sm.Scan(0,st).NParam()) = sm.Scan(s,st).GetParams().t();
  }

  if (clo.WriteSliceStats()) {
    char istring[256];
    if (st==EDDY::DWI) sprintf(istring,"%s.EddyDwiSliceStatsIteration%02d",clo.IOutFname().c_str(),iter);
    else sprintf(istring,"%s.Eddyb0SliceStatsIteration%02d",clo.IOutFname().c_str(),iter);
    stats.Write(string(istring));
    rm.DumpOutlierMaps(string(istring));
  }
} EddyCatch

void AddRotation(ECScanManager&               sm,
                 const NEWMAT::ColumnVector&  rp) EddyTry
{
  for (unsigned int i=0; i<sm.NScans(); i++) {
    NEWMAT::ColumnVector mp = sm.Scan(i).GetParams(EDDY::MOVEMENT);
    mp(4) += rp(1); mp(5) += rp(2); mp(6) += rp(3);
    sm.Scan(i).SetParams(mp,EDDY::MOVEMENT);
  }
} EddyCatch

void PrintMIValues(const EddyCommandLineOptions&  clo,
                   const ECScanManager&           sm,
                   const std::string&             fname,
                   bool                           write_planes) EddyTry
{
  std::vector<std::string> dir(6);
  dir[0]="xt"; dir[1]="yt"; dir[2]="zt";
  dir[3]="xr"; dir[4]="yr"; dir[5]="zr";
  // First write 1D profiles along main directions
  for (unsigned int i=0; i<6; i++) {
    std::vector<unsigned int> n(6,1); n[i] = 100;
    std::vector<double> first(6,0.0); first[i] = -2.5;
    std::vector<double> last(6,0.0); last[i] = 2.5;
    if (clo.VeryVerbose()) cout << "Writing MI values for direction " << i << endl;
    PEASUtils::WritePostEddyBetweenShellMIValues(clo,sm,n,first,last,fname+"_"+dir[i]);
  }
  // Write 2D planes if requested
  for (unsigned int i=0; i<6; i++) {
    for (unsigned int j=i+1; j<6; j++) {
      std::vector<unsigned int> n(6,1); n[i] = 20; n[j] = 20;
      std::vector<double> first(6,0.0); first[i] = -1.0; first[j] = -1.0;
      std::vector<double> last(6,0.0); last[i] = 1.0; last[j] = 1.0;
      if (clo.VeryVerbose()) cout << "Writing MI values for plane " << i << "-" << j << endl;
      PEASUtils::WritePostEddyBetweenShellMIValues(clo,sm,n,first,last,fname+"_"+dir[i]+"_"+dir[j]);
    }
  }
} EddyCatch

} // End namespace EDDY

/*! \mainpage
 * Here goes a description of the eddy project
 */
