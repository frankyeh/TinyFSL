// Declarations of classes that implements useful
// utility functions for the eddy current project.
// They are collections of statically declared
// functions that have been collected into classes
// to make it explicit where they come from. There
// will never be any instances of theses classes.
//
// EddyUtils.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford
//

#include <cstdlib>
#include <string>

#include <vector>
#include <cfloat>
#include <cmath>
#include "armawrap/newmat.h"
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS           // To allow us to use .set_sform etc
#endif
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "cprob/libprob.h"
#include "warpfns/warpfns.h"
#include "EddyHelperClasses.h"
#include "EddyUtils.h"

using namespace std;
using namespace CPROB;
using namespace EDDY;

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class EddyUtils
//
// Helper Class used to perform various useful tasks for
// the eddy current correction project.
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

bool EddyUtils::get_groups(// Input
			   const std::vector<DiffPara>&             dpv,
			   // Output
			   std::vector<std::vector<unsigned int> >& grps,
			   std::vector<unsigned int>&               grpi,
			   std::vector<double>&                     grpb) EddyTry
{
  std::vector<unsigned int>     grp_templates;
  // First pass to sort out how many different b-values/shells there are
  grp_templates.push_back(0);
  for (unsigned int i=1; i<dpv.size(); i++) {
    unsigned int j;
    for (j=0; j<grp_templates.size(); j++) { if (EddyUtils::AreInSameShell(dpv[grp_templates[j]],dpv[i])) break; }
    if (j == grp_templates.size()) grp_templates.push_back(i);
  }
  // Second pass to centre the b-values in each group
  grpb.resize(grp_templates.size());
  std::vector<unsigned int>   grp_n(grp_templates.size(),1);
  for (unsigned int j=0; j<grp_templates.size(); j++) {
    grpb[j] = dpv[grp_templates[j]].bVal();
    for (unsigned int i=0; i<dpv.size(); i++) {
      if (EddyUtils::AreInSameShell(dpv[grp_templates[j]],dpv[i]) && i!=grp_templates[j]) {
	grpb[j] += dpv[i].bVal();
	grp_n[j]++;
      }
    }
    grpb[j] /= grp_n[j];
  }
  // Sort groups by ascending b-values
  std::sort(grpb.begin(),grpb.end());
  // Second pass to assign groups based on mean b-values
  grpi.resize(dpv.size()); grps.resize(grpb.size());
  for (unsigned int j=0; j<grpb.size(); j++) {
    grp_n[j] = 0;
    for (unsigned int i=0; i<dpv.size(); i++) {
      if (std::abs(dpv[i].bVal()-grpb[j]) <= EddyUtils::b_range) { grpi[i] = j; grps[j].push_back(i); grp_n[j]++; }
    }
  }
  // Check to see if it is plausible that it is shelled data
  bool is_shelled;
  if (EddyUtils::Isb0(EDDY::DiffPara(grpb[0]))) { // If it includes "b0"-shell
    is_shelled = grpb.size() < 7; // Don't trust more than 5 shells
    unsigned int scans_per_shell = static_cast<unsigned int>((double(dpv.size() - grp_n[0]) / double(grpb.size() - 1)) + 0.5);
    is_shelled &= bool(*std::max_element(grp_n.begin()+1,grp_n.end()) < 2 * scans_per_shell); // Don't trust too many scans in one shell
    is_shelled &= bool(3 * *std::min_element(grp_n.begin()+1,grp_n.end()) > scans_per_shell); // Don't trust too few scans in one shell
  }
  else { // If all scans are dwis
    is_shelled = grpb.size() < 6; // Don't trust more than 5 shells
    unsigned int scans_per_shell = static_cast<unsigned int>((double(dpv.size()) / double(grpb.size())) + 0.5);
    is_shelled &= bool(*std::max_element(grp_n.begin(),grp_n.end()) < 2 * scans_per_shell); // Don't trust too many scans in one shell
    is_shelled &= bool(3 * *std::min_element(grp_n.begin(),grp_n.end()) > scans_per_shell); // Don't trust too few scans in one shell
  }
  if (!is_shelled) return(false);
  // Final sanity check
  unsigned int nscan = grps[0].size();
  for (unsigned int i=1; i<grps.size(); i++) nscan += grps[i].size();
  if (nscan != dpv.size()) throw EddyException("EddyUtils::get_groups: Inconsistent b-values detected");
  return(true);
} EddyCatch

bool EddyUtils::IsShelled(const std::vector<DiffPara>& dpv) EddyTry
{
  std::vector<std::vector<unsigned int> > grps;
  std::vector<unsigned int>               grpi;
  std::vector<double>                     grpb;
  return(get_groups(dpv,grps,grpi,grpb));
} EddyCatch

bool EddyUtils::IsMultiShell(const std::vector<DiffPara>& dpv) EddyTry
{
  std::vector<std::vector<unsigned int> > grps;
  std::vector<unsigned int>               grpi;
  std::vector<double>                     grpb;
  bool is_shelled = get_groups(dpv,grps,grpi,grpb);
  return(is_shelled && grpb.size() > 1);
} EddyCatch

bool EddyUtils::GetGroups(// Input
			   const std::vector<DiffPara>&             dpv,
			   // Output
			   std::vector<unsigned int>&               grpi,
			   std::vector<double>&                     grpb) EddyTry
{
  std::vector<std::vector<unsigned int> > grps;
  return(get_groups(dpv,grps,grpi,grpb));
} EddyCatch

bool EddyUtils::GetGroups(// Input
			   const std::vector<DiffPara>&             dpv,
			   // Output
			   std::vector<std::vector<unsigned int> >& grps,
			   std::vector<double>&                     grpb) EddyTry
{
  std::vector<unsigned int> grpi;
  return(get_groups(dpv,grps,grpi,grpb));
} EddyCatch

bool EddyUtils::GetGroups(// Input
			   const std::vector<DiffPara>&             dpv,
			   // Output
			   std::vector<std::vector<unsigned int> >& grps,
			   std::vector<unsigned int>&               grpi,
			   std::vector<double>&                     grpb) EddyTry
{
  return(get_groups(dpv,grps,grpi,grpb));
} EddyCatch

std::vector<unsigned int> EddyUtils::GetIndiciesOfDWIs(const std::vector<DiffPara>& dpars) EddyTry
{
  std::vector<unsigned int> indicies;
  for (unsigned int i=0; i<dpars.size(); i++) { if (EddyUtils::IsDiffusionWeighted(dpars[i])) indicies.push_back(i); }
  return(indicies);
} EddyCatch

std::vector<DiffPara> EddyUtils::GetDWIDiffParas(const std::vector<DiffPara>&   dpars) EddyTry
{
  std::vector<unsigned int> indx = EddyUtils::GetIndiciesOfDWIs(dpars);
  std::vector<DiffPara> dwi_dpars;
  for (unsigned int i=0; i<indx.size(); i++) dwi_dpars.push_back(dpars[indx[i]]);
  return(dwi_dpars);
} EddyCatch

bool EddyUtils::AreMatchingPair(const ECScan& s1, const ECScan& s2) EddyTry
{
  double dp = NEWMAT::DotProduct(s1.GetAcqPara().PhaseEncodeVector(),s2.GetAcqPara().PhaseEncodeVector());
  if (std::abs(dp + 1.0) > 1e-6) return(false);
  if (!EddyUtils::AreInSameShell(s1.GetDiffPara(),s2.GetDiffPara())) return(false);
  if (IsDiffusionWeighted(s1.GetDiffPara()) && !HaveSameDirection(s1.GetDiffPara(),s2.GetDiffPara())) return(false);
  return(true);
} EddyCatch

std::vector<NEWMAT::Matrix> EddyUtils::GetSliceWiseForwardMovementMatrices(const EDDY::ECScan&           scan) EddyTry
{
  std::vector<NEWMAT::Matrix> R(scan.GetIma().zsize());
  for (unsigned int tp=0; tp<scan.GetMBG().NGroups(); tp++) {
    NEWMAT::Matrix tR = scan.ForwardMovementMatrix(tp);
    std::vector<unsigned int> slices = scan.GetMBG().SlicesAtTimePoint(tp);
    for (unsigned int i=0; i<slices.size(); i++) R[slices[i]] = tR;
  }
  return(R);
} EddyCatch

std::vector<NEWMAT::Matrix> EddyUtils::GetSliceWiseInverseMovementMatrices(const EDDY::ECScan&           scan) EddyTry
{
  std::vector<NEWMAT::Matrix> R(scan.GetIma().zsize());
  for (unsigned int tp=0; tp<scan.GetMBG().NGroups(); tp++) {
    NEWMAT::Matrix tR = scan.InverseMovementMatrix(tp);
    std::vector<unsigned int> slices = scan.GetMBG().SlicesAtTimePoint(tp);
    for (unsigned int i=0; i<slices.size(); i++) R[slices[i]] = tR;
  }
  return(R);
} EddyCatch

int EddyUtils::read_DWI_volume4D(NEWIMAGE::volume4D<float>&     dwivols,
				 const std::string&             fname,
				 const std::vector<DiffPara>&   dpars) EddyTry
{
  std::vector<unsigned int> indx = EddyUtils::GetIndiciesOfDWIs(dpars);
  NEWIMAGE::volume<float> tmp;
  read_volumeROI(tmp,fname,0,0,0,0,-1,-1,-1,0);
  dwivols.reinitialize(tmp.xsize(),tmp.ysize(),tmp.zsize(),indx.size());
  for (unsigned int i=0; i<indx.size(); i++) {
    read_volumeROI(tmp,fname,0,0,0,indx[i],-1,-1,-1,indx[i]);
    dwivols[i] = tmp;
  }
  return(1);
} EddyCatch

NEWIMAGE::volume<float> EddyUtils::ConvertMaskToFloat(const NEWIMAGE::volume<char>& charmask) EddyTry
{
  NEWIMAGE::volume<float> floatmask(charmask.xsize(),charmask.ysize(),charmask.zsize());
  NEWIMAGE::copybasicproperties(charmask,floatmask);
  for (int k=0; k<charmask.zsize(); k++) {
    for (int j=0; j<charmask.ysize(); j++) {
      for (int i=0; i<charmask.xsize(); i++) {
	floatmask(i,j,k) = static_cast<float>(charmask(i,j,k));
      }
    }
  }
  return(floatmask);
} EddyCatch

// Rewritten for new newimage
NEWIMAGE::volume<float> EddyUtils::Smooth(const NEWIMAGE::volume<float>& ima, float fwhm, const NEWIMAGE::volume<float>& mask) EddyTry
{
  if (mask.getextrapolationmethod() != NEWIMAGE::zeropad) throw EddyException("EddyUtils::Smooth: mask must use zeropad for extrapolation");
  float sx = (fwhm/std::sqrt(8.0*std::log(2.0)))/ima.xdim();
  float sy = (fwhm/std::sqrt(8.0*std::log(2.0)))/ima.ydim();
  float sz = (fwhm/std::sqrt(8.0*std::log(2.0)))/ima.zdim();
  int nx=((int) (sx-0.001))*2 + 3;
  int ny=((int) (sy-0.001))*2 + 3;
  int nz=((int) (sz-0.001))*2 + 3;
  NEWMAT::ColumnVector krnlx = NEWIMAGE::gaussian_kernel1D(sx,nx);
  NEWMAT::ColumnVector krnly = NEWIMAGE::gaussian_kernel1D(sy,ny);
  NEWMAT::ColumnVector krnlz = NEWIMAGE::gaussian_kernel1D(sz,nz);
  NEWIMAGE::volume4D<float> ovol = ima; // volume4D just an alias for volume
  for (int i=0; i<ima.tsize(); i++) {
    ovol[i] = NEWIMAGE::convolve_separable(ima[i],krnlx,krnly,krnlz,mask)*mask;
  }
  return(ovol);
} EddyCatch

// Made obsolete by new newimage rewrite
/*
NEWIMAGE::volume4D<float> EddyUtils::Smooth4D(const NEWIMAGE::volume4D<float>& ima, float fwhm, const NEWIMAGE::volume<float>&   mask)
{
  NEWIMAGE::volume4D<float> ovol = ima;
  for (int i=0; i<ima.tsize(); i++) {
    ovol[i] = EddyUtils::Smooth3D(ima[i],fwhm,mask);
  }
  return(ovol);
}
*/

NEWIMAGE::volume<float> EddyUtils::MakeNoiseIma(const NEWIMAGE::volume<float>& ima, float mu, float stdev) EddyTry
{
  NEWIMAGE::volume<float>  nima = ima;
  double rnd;
  for (int k=0; k<nima.zsize(); k++) {
    for (int j=0; j<nima.ysize(); j++) {
      for (int i=0; i<nima.xsize(); i++) {
	drand(&rnd);
	nima(i,j,k) = mu + stdev*static_cast<float>(ndtri(rnd-1));
      }
    }
  }
  return(nima);
} EddyCatch

DiffStats EddyUtils::GetSliceWiseStats(// Input
				       const NEWIMAGE::volume<float>&                  pred,          // Prediction in model space
				       std::shared_ptr<const NEWIMAGE::volume<float> > susc,          // Susceptibility induced off-resonance field
				       const NEWIMAGE::volume<float>&                  pmask,         // "Data valid" mask in model space
				       const NEWIMAGE::volume<float>&                  bmask,         // Brain mask in model space
				       const EDDY::ECScan&                             scan) EddyTry  // Scan we want to register to pred
{
  // Transform prediction into observation space
  NEWIMAGE::volume<float> pios = EddyUtils::TransformModelToScanSpace(pred,scan,susc);
  // Transform binary mask into observation space
  NEWIMAGE::volume<float> mask = pred; mask = 0.0;
  NEWIMAGE::volume<float> bios = EddyUtils::transform_model_to_scan_space(pmask*bmask,scan,susc,false,mask,NULL,NULL);
  bios.binarise(0.99); // Value above (arbitrary) 0.99 implies valid voxels
  mask *= bios; // Volume and prediction mask falls within FOV
  // Calculate slice-wise stats from difference image
  DiffStats stats(scan.GetOriginalIma()-pios,mask);
  return(stats);
} EddyCatch

double EddyUtils::param_update(// Input
			       const NEWIMAGE::volume<float>&                    pred,      // Prediction in model space
			       std::shared_ptr<const NEWIMAGE::volume<float> >   susc,      // Susceptibility off-resonance field
			       std::shared_ptr<const NEWIMAGE::volume<float> >   bias,      // Recieve bias field
			       const NEWIMAGE::volume<float>&                    pmask,     // "Data valid" mask in model space
			       Parameters                                        whichp,    // Which parameters to update
			       bool                                              cbs,       // Check (success of parameters) Before Set
			       float                                             fwhm,      // FWHM for Gaussian smoothing
			       // These input parameters are for debugging only
			       unsigned int                                      scindx,    // Scan index
			       unsigned int                                      iter,      // Iteration
			       unsigned int                                      level,     // Determines how much gets written
			       // Input/output
			       EDDY::ECScan&                                     scan)      // Scan we want to register to pred
EddyTry
{
  // Transform prediction into observation space
  NEWIMAGE::volume<float> mask = pred; mask.setextrapolationmethod(NEWIMAGE::zeropad); mask = 0.0;
  NEWIMAGE::volume<float> jac = pred; jac = 1.0;
  NEWIMAGE::volume<float> pios = EddyUtils::transform_model_to_scan_space(pred,scan,susc,true,mask,&jac,NULL);
  // NEWIMAGE::volume<float> pios = EddyUtils::transform_model_to_scan_space(pred,scan,susc,true,mask,&jac,NULL,scindx,iter,level);
  // Transform binary mask into observation space
  NEWIMAGE::volume<float> skrutt = pred; skrutt = 0.0;
  NEWIMAGE::volume<float> mios = EddyUtils::transform_model_to_scan_space(pmask,scan,susc,false,skrutt,NULL,NULL);
  mios.binarise(0.99); // Value above (arbitrary) 0.99 implies valid voxels
  mask *= mios; // Volume and prediction mask falls within FOV
  // Get partial derivatives w.r.t. to requested category of parameters in prediction space
  NEWIMAGE::volume4D<float> derivs = EddyUtils::get_partial_derivatives_in_scan_space(pred,scan,susc,whichp);
  if (fwhm) { mask.setextrapolationmethod(NEWIMAGE::zeropad); derivs = EddyUtils::Smooth(derivs,fwhm,mask); }
  // Calculate XtX where X is a matrix whos columns are the partial derivatives
  NEWMAT::Matrix XtX = EddyUtils::make_XtX(derivs,mask);
  // Calculate difference image between observed and predicted
  NEWIMAGE::volume<float> dima = pios-scan.GetIma();
  if (fwhm) { mask.setextrapolationmethod(NEWIMAGE::zeropad); dima = EddyUtils::Smooth(dima,fwhm,mask); }
  // Calculate Xty where y is the difference between observed and predicted. X as above.
  NEWMAT::ColumnVector Xty = EddyUtils::make_Xty(derivs,dima,mask);
  // Get derivative and Hessian of regularisation (relevant only for slice-to-vol);
  NEWMAT::ColumnVector lHb = scan.GetRegGrad(whichp);
  NEWMAT::Matrix H = scan.GetRegHess(whichp);
  // Calculate mean sum of squares from difference image and add regularisation
  double masksum = mask.sum();
  double mss = (dima*mask).sumsquares() / masksum + scan.GetReg(whichp);
  // Very mild Tikhonov regularisation to select solution with smaller norm
  double lambda = 1.0/masksum;
  NEWMAT::IdentityMatrix eye(XtX.Nrows());
  // Calculate update to parameters
  // NEWMAT::ColumnVector old_style_update = -XtX.i()*Xty;
  NEWMAT::ColumnVector update = -(XtX/masksum + H + lambda*eye).i()*(Xty/masksum + lHb);
  // Calculate sims (scan in model space) if we need to write it as debug info
  NEWIMAGE::volume<float> sims;
  if (level) sims = scan.GetUnwarpedIma(susc);
  // Update parameters
  for (unsigned int i=0; i<scan.NDerivs(whichp); i++) {
    scan.SetDerivParam(i,scan.GetDerivParam(i,whichp)+update(i+1),whichp);
  }

  // Check if update makes sense, and write warning if it doesn't and debug is set
  if (level && !EddyUtils::UpdateMakesSense(scan,update)) {
    cout << "EddyUtils::param_update: update doesn't make sense" << endl;
  }
  // We need to write debug info before we check if parameter update improved things
  if (level) EddyUtils::write_debug_info_for_param_update(scan,scindx,iter,level,cbs,fwhm,derivs,mask,mios,pios,
							  jac,susc,bias,pred,dima,sims,pmask,XtX,Xty,update);

  // Check if parameters actually improved things and reject them if not
  if (cbs) {
    pios = EddyUtils::TransformModelToScanSpace(pred,scan,susc);
    // Transform binary mask into observation space
    mask = 0.0;
    mios = EddyUtils::transform_model_to_scan_space(pmask,scan,susc,false,mask,NULL,NULL);
    mios.binarise(0.99); // Value above (arbitrary) 0.99 implies valid voxels
    mask *= mios; // Volume and prediction mask falls within FOV
    dima = pios-scan.GetIma();
    if (fwhm) { mask.setextrapolationmethod(NEWIMAGE::zeropad); dima = EddyUtils::Smooth(dima,fwhm,mask); }
    double mss_au = ((dima*mask).sumsquares()) / mask.sum();
    if (std::isnan(mss_au) || mss_au > mss) { // If cost not decreased, set parameters back to what they were
      for (unsigned int i=0; i<scan.NDerivs(whichp); i++) {
	scan.SetDerivParam(i,scan.GetDerivParam(i,whichp)-update(i+1),whichp);
      }
      if (level) { // Write out info about failed update if it is a debug run
	cout << "EddyInternalGpuUtils::param_update: updates rejected" << endl;
	cout << "EddyInternalGpuUtils::param_update: original mss = " << mss << ", after update mss = " << mss_au << endl;
	cout.flush();
      }
    }
  }
  return(mss);
} EddyCatch

void EddyUtils::write_debug_info_for_param_update(const EDDY::ECScan&                                scan,
						  unsigned int                                       scindx,
						  unsigned int                                       iter,
						  unsigned int                                       level,
						  bool                                               cbs,
						  float                                              fwhm,
						  const NEWIMAGE::volume4D<float>&                   derivs,
						  const NEWIMAGE::volume<float>&                     mask,
						  const NEWIMAGE::volume<float>&                     mios,
						  const NEWIMAGE::volume<float>&                     pios,
						  const NEWIMAGE::volume<float>&                     jac,
						  std::shared_ptr<const NEWIMAGE::volume<float> >    susc,
						  std::shared_ptr<const NEWIMAGE::volume<float> >    bias,
						  const NEWIMAGE::volume<float>&                     pred,
						  const NEWIMAGE::volume<float>&                     dima,
						  const NEWIMAGE::volume<float>&                     sims,
						  const NEWIMAGE::volume<float>&                     pmask,
						  const NEWMAT::Matrix&                              XtX,
						  const NEWMAT::ColumnVector&                        Xty,
						  const NEWMAT::ColumnVector&                        update) EddyTry
{
  char fname[256], bname[256];
  if (scan.IsSliceToVol()) strcpy(bname,"EDDY_DEBUG_S2V");
  else strcpy(bname,"EDDY_DEBUG");
  if (level>0) {
    sprintf(fname,"%s_masked_dima_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(dima*mask,fname);
    sprintf(fname,"%s_reverse_dima_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(pred-sims,fname);
  }
  if (level>1) {
    sprintf(fname,"%s_mask_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(mask,fname);
    sprintf(fname,"%s_pios_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(pios,fname);
    sprintf(fname,"%s_pred_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(pred,fname);
    sprintf(fname,"%s_dima_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(dima,fname);
    sprintf(fname,"%s_jac_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(jac,fname);
    sprintf(fname,"%s_orig_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(scan.GetIma(),fname);
    if (cbs) {
      NEWIMAGE::volume<float> new_pios = EddyUtils::TransformModelToScanSpace(pred,scan,susc);
      NEWIMAGE::volume<float> new_mask = new_pios; new_mask = 0.0;
      NEWIMAGE::volume<float> new_mios = EddyUtils::transform_model_to_scan_space(pmask,scan,susc,false,new_mask,NULL,NULL);
      new_mios.binarise(0.99); // Value above (arbitrary) 0.99 implies valid voxels
      new_mask *= new_mios; // Volume and prediction mask falls within FOV
      NEWIMAGE::volume<float> new_dima = new_pios-scan.GetIma();
      if (fwhm) { mask.setextrapolationmethod(NEWIMAGE::zeropad); new_dima = EddyUtils::Smooth(new_dima,fwhm,mask); }
      sprintf(fname,"%s_new_masked_dima_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(new_dima*new_mask,fname);
      NEWIMAGE::volume<float> new_sims = scan.GetUnwarpedIma(susc);
      sprintf(fname,"%s_new_reverse_dima_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(pred-new_sims,fname);
      sprintf(fname,"%s_new_mask_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(new_mask,fname);
      sprintf(fname,"%s_new_pios_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(new_pios,fname);
      sprintf(fname,"%s_new_dima_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(new_dima,fname);
    }
  }
  if (level>2) {
    sprintf(fname,"%s_mios_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(mios,fname);
    sprintf(fname,"%s_pmask_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(pmask,fname);
    sprintf(fname,"%s_derivs_%02d_%04d",bname,iter,scindx); NEWIMAGE::write_volume(derivs,fname);
    sprintf(fname,"%s_XtX_%02d_%04d.txt",bname,iter,scindx); MISCMATHS::write_ascii_matrix(fname,XtX);
    sprintf(fname,"%s_Xty_%02d_%04d.txt",bname,iter,scindx); MISCMATHS::write_ascii_matrix(fname,Xty);
    sprintf(fname,"%s_update_%02d_%04d.txt",bname,iter,scindx); MISCMATHS::write_ascii_matrix(fname,update);
  }
  return;
} EddyCatch
/*
double EddyUtils::param_update(// Input
			       const NEWIMAGE::volume<float>&                      pred,      // Prediction in model space
			       boost::shared_ptr<const NEWIMAGE::volume<float> >   susc,      // Susceptibility induced off-resonance field
			       const NEWIMAGE::volume<float>&                      pmask,     //
			       Parameters                                          whichp,    // What parameters do we want to update
			       bool                                                cbs,       // Check (success of parameters) Before Set
			       // Input/output
			       EDDY::ECScan&                                       scan,      // Scan we want to register to pred
			       // Output
			       NEWMAT::ColumnVector                                *rupdate)  // Vector of updates, optional output
{
  // Transform prediction into observation space
  NEWIMAGE::volume<float> pios = EddyUtils::TransformModelToScanSpace(pred,scan,susc);
  // Transform binary mask into observation space
  NEWIMAGE::volume<float> mask = pred; mask = 0.0;
  NEWIMAGE::volume<float> bios = EddyUtils::transform_model_to_scan_space(pmask,scan,susc,false,mask,NULL,NULL);
  bios.binarise(0.99); // Value above (arbitrary) 0.99 implies valid voxels
  mask *= bios;        // Volume and prediction mask falls within FOV
  // Get partial derivatives w.r.t. requested category of parameters in prediction space
  NEWIMAGE::volume4D<float> derivs = EddyUtils::get_partial_derivatives_in_scan_space(pred,scan,susc,whichp);
  // Calculate XtX where X is a matrix whos columns are the partial derivatives
  NEWMAT::Matrix XtX = EddyUtils::make_XtX(derivs,mask);
  // Calculate difference image between observed and predicted
  NEWIMAGE::volume<float> dima = pios-scan.GetIma();
  // Calculate Xty where y is the difference between observed and predicted. X as above.
  NEWMAT::ColumnVector Xty = EddyUtils::make_Xty(derivs,dima,mask);
  // Calculate mean sum of squares from difference image
  double mss = (dima*mask).sumsquares() / mask.sum();
  // Calculate update to parameters
  NEWMAT::ColumnVector update = -XtX.i()*Xty;
  // Update parameters
  for (unsigned int i=0; i<scan.NDerivs(whichp); i++) {
    scan.SetDerivParam(i,scan.GetDerivParam(i,whichp)+update(i+1),whichp);
  }
  if (cbs) {
    pios = EddyUtils::TransformModelToScanSpace(pred,scan,susc);
    // Transform binary mask into observation space
    mask = 0.0;
    bios = EddyUtils::transform_model_to_scan_space(pmask,scan,susc,false,mask,NULL,NULL);
    bios.binarise(0.99); // Value above (arbitrary) 0.99 implies valid voxels
    mask *= bios; // Volume and prediction mask falls within FOV
    double mss_au = (((pios-scan.GetIma())*mask).sumsquares()) / mask.sum();
    if (mss_au > mss) { // Oh dear
      for (unsigned int i=0; i<scan.NDerivs(whichp); i++) {
	scan.SetDerivParam(i,scan.GetDerivParam(i,whichp)-update(i+1),whichp);
      }
    }
  }
  if (rupdate) *rupdate = update;
  return(mss);
}

*/

NEWIMAGE::volume<float> EddyUtils::transform_model_to_scan_space(// Input
								 const NEWIMAGE::volume<float>&                    pred,
								 const EDDY::ECScan&                               scan,
								 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
								 bool                                              jacmod,
								 // Output
								 NEWIMAGE::volume<float>&                          omask,
								 NEWIMAGE::volume<float>                           *jac,
								 NEWIMAGE::volume4D<float>                         *grad) EddyTry
{
  // Get total field from scan
  if (jacmod && !jac) throw EddyException("EddyUtils::transform_model_to_scan_space: jacmod can only be used with valid jac");
  NEWIMAGE::volume4D<float> dfield;
  if (jacmod || jac) dfield = scan.FieldForModelToScanTransform(susc,omask,*jac);
  else dfield = scan.FieldForModelToScanTransform(susc,omask);
  NEWMAT::Matrix eye(4,4); eye=0; eye(1,1)=1.0; eye(2,2)=1.0; eye(3,3)=1.0; eye(4,4)=1.0;
  NEWIMAGE::volume<float> ovol = pred; ovol = 0.0;
  NEWIMAGE::volume<char> mask3(pred.xsize(),pred.ysize(),pred.zsize());
  NEWIMAGE::copybasicproperties(pred,mask3); mask3 = 0;
  std::vector<int> ddir(3); ddir[0] = 0; ddir[1] = 1; ddir[2] = 2;
  if (scan.IsSliceToVol()) {
    if (grad) {
      grad->reinitialize(pred.xsize(),pred.ysize(),pred.zsize(),3);
      NEWIMAGE::copybasicproperties(pred,*grad);
    }
    for (unsigned int tp=0; tp<scan.GetMBG().NGroups(); tp++) { // tp for timepoint
      NEWMAT::Matrix R = scan.ForwardMovementMatrix(tp);
      std::vector<unsigned int> slices = scan.GetMBG().SlicesAtTimePoint(tp);
      if (grad) NEWIMAGE::raw_general_transform(pred,eye,dfield,ddir,ddir,slices,&eye,&R,ovol,*grad,&mask3);
      else NEWIMAGE::apply_warp(pred,eye,dfield,eye,R,slices,ovol,mask3);
    }
  }
  else {
    std::vector<unsigned int> all_slices;
    // Get RB matrix
    NEWMAT::Matrix R = scan.ForwardMovementMatrix();
    // Transform prediction using RB, inverted Tot map and Jacobian
    if (grad) {
      grad->reinitialize(pred.xsize(),pred.ysize(),pred.zsize(),3);
      NEWIMAGE::copybasicproperties(pred,*grad);
      NEWIMAGE::raw_general_transform(pred,eye,dfield,ddir,ddir,all_slices,&eye,&R,ovol,*grad,&mask3);
    }
    else NEWIMAGE::apply_warp(pred,eye,dfield,eye,R,ovol,mask3);
  }
  omask *= EddyUtils::ConvertMaskToFloat(mask3); // Combine all masks
  EddyUtils::SetTrilinearInterp(omask);
  if (jacmod) ovol *= *jac;                      // Jacobian modulation if it was asked for
  return(ovol);
} EddyCatch

// This is a temporary version to allow for writing of debug information
/*
NEWIMAGE::volume<float> EddyUtils::transform_model_to_scan_space(// Input
								 const NEWIMAGE::volume<float>&                    pred,
								 const EDDY::ECScan&                               scan,
								 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
								 bool                                              jacmod,
								 // Output
								 NEWIMAGE::volume<float>&                          omask,
								 NEWIMAGE::volume<float>                           *jac,
								 NEWIMAGE::volume4D<float>                         *grad,
								 // Tmp
								 unsigned int                                      scindx,
								 unsigned int                                      iter,
								 unsigned int                                      level) EddyTry
{
  // Get total field from scan
  if (jacmod && !jac) throw EddyException("EddyUtils::transform_model_to_scan_space: jacmod can only be used with valid jac");
  NEWIMAGE::volume4D<float> dfield;
  if (jacmod || jac) dfield = scan.FieldForModelToScanTransform(susc,omask,*jac);
  else dfield = scan.FieldForModelToScanTransform(susc,omask);
  NEWMAT::Matrix eye(4,4); eye=0; eye(1,1)=1.0; eye(2,2)=1.0; eye(3,3)=1.0; eye(4,4)=1.0;
  NEWIMAGE::volume<float> ovol = pred; ovol = 0.0;
  NEWIMAGE::volume<char> mask3(pred.xsize(),pred.ysize(),pred.zsize());
  NEWIMAGE::copybasicproperties(pred,mask3); mask3 = 0;
  std::vector<int> ddir(3); ddir[0] = 0; ddir[1] = 1; ddir[2] = 2;
  if (scan.IsSliceToVol()) {
    if (grad) {
      grad->reinitialize(pred.xsize(),pred.ysize(),pred.zsize(),3);
      NEWIMAGE::copybasicproperties(pred,*grad);
    }
    for (unsigned int tp=0; tp<scan.GetMBG().NGroups(); tp++) { // tp for timepoint
      NEWMAT::Matrix R = scan.ForwardMovementMatrix(tp);
      std::vector<unsigned int> slices = scan.GetMBG().SlicesAtTimePoint(tp);
      if (grad) NEWIMAGE::raw_general_transform(pred,eye,dfield,ddir,ddir,slices,&eye,&R,ovol,*grad,&mask3);
      else NEWIMAGE::apply_warp(pred,eye,dfield,eye,R,slices,ovol,mask3);
    }
  }
  else {
    std::vector<unsigned int> all_slices;
    // Get RB matrix
    NEWMAT::Matrix R = scan.ForwardMovementMatrix();
    // Transform prediction using RB, inverted Tot map and Jacobian
    if (grad) {
      grad->reinitialize(pred.xsize(),pred.ysize(),pred.zsize(),3);
      NEWIMAGE::copybasicproperties(pred,*grad);
      NEWIMAGE::raw_general_transform(pred,eye,dfield,ddir,ddir,all_slices,&eye,&R,ovol,*grad,&mask3);
    }
    else NEWIMAGE::apply_warp(pred,eye,dfield,eye,R,ovol,mask3);
  }
  // mask3 is buggered already here
  char bfname[256];
  char fname[256];
  if (level) {
    if (scan.IsSliceToVol()) strcpy(bfname,"EDDY_DEBUG_SPECIAL_S2V");
    else strcpy(bfname,"EDDY_DEBUG_SPECIAL");
    sprintf(fname,"%s_omask_before_%02d_%04d",bfname,iter,scindx); NEWIMAGE::write_volume(omask,fname);
    sprintf(fname,"%s_mask3_before_%02d_%04d",bfname,iter,scindx); NEWIMAGE::write_volume(mask3,fname);
  }
  omask *= EddyUtils::ConvertMaskToFloat(mask3); // Combine all masks
  if (level) { sprintf(fname,"%s_omask_after_1_%02d_%04d",bfname,iter,scindx); NEWIMAGE::write_volume(omask,fname); }
  EddyUtils::SetTrilinearInterp(omask);
  if (level) { sprintf(fname,"%s_omask_after_2_%02d_%04d",bfname,iter,scindx); NEWIMAGE::write_volume(omask,fname); }
  if (jacmod) ovol *= *jac;                      // Jacobian modulation if it was asked for
  return(ovol);
} EddyCatch
*/

// Has been modified for slice-to-vol
EDDY::ImageCoordinates EddyUtils::transform_coordinates_from_model_to_scan_space(// Input
										 const NEWIMAGE::volume<float>&                    pred,
										 const EDDY::ECScan&                               scan,
										 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
										 // Output
										 NEWIMAGE::volume<float>                           *omask,
										 NEWIMAGE::volume<float>                           *jac) EddyTry
{
  // Get total field from scan
  NEWIMAGE::volume4D<float> dfield;
  if (omask && jac) dfield = scan.FieldForModelToScanTransform(susc,*omask,*jac);
  else if (omask) dfield = scan.FieldForModelToScanTransform(susc,*omask);
  else if (jac) dfield = scan.FieldForModelToScanTransformWithJac(susc,*jac);
  else dfield = scan.FieldForModelToScanTransform(susc);

  ImageCoordinates coord(pred.xsize(),pred.ysize(),pred.zsize());
  if (scan.IsSliceToVol()) {
    for (unsigned int tp=0; tp<scan.GetMBG().NGroups(); tp++) { // tp for timepoint
      NEWMAT::Matrix R = scan.ForwardMovementMatrix(tp);
      std::vector<unsigned int> slices = scan.GetMBG().SlicesAtTimePoint(tp);
      // Transform coordinates using RB and inverted Tot map
      if (omask) {
	NEWIMAGE::volume<float> mask2(pred.xsize(),pred.ysize(),pred.zsize());
	NEWIMAGE::copybasicproperties(pred,mask2); mask2 = 0;
	EddyUtils::transform_coordinates(pred,dfield,R,slices,coord,&mask2);
	*omask *= mask2;
	EddyUtils::SetTrilinearInterp(*omask);
      }
      else EddyUtils::transform_coordinates(pred,dfield,R,slices,coord,NULL);
    }
  }
  else {
  // Get RB matrix
    NEWMAT::Matrix R = scan.ForwardMovementMatrix();
    std::vector<unsigned int> all_slices;
    // Transform coordinates using RB and inverted Tot map
    if (omask) {
      NEWIMAGE::volume<float> mask2(pred.xsize(),pred.ysize(),pred.zsize());
      NEWIMAGE::copybasicproperties(pred,mask2); mask2 = 0;
      EddyUtils::transform_coordinates(pred,dfield,R,all_slices,coord,&mask2);
      *omask *= mask2;
      EddyUtils::SetTrilinearInterp(*omask);
    }
    else EddyUtils::transform_coordinates(pred,dfield,R,all_slices,coord,NULL);
  }

  return(coord);
} EddyCatch

// Has been modified for slice-to-vol

NEWIMAGE::volume4D<float> EddyUtils::get_partial_derivatives_in_scan_space(// Input
									   const NEWIMAGE::volume<float>&                    pred,      // Prediction in model space
									   const EDDY::ECScan&                               scan,      // Scan space
									   std::shared_ptr<const NEWIMAGE::volume<float> >   susc,      // Susceptibility off-resonance field
									   EDDY::Parameters                                  whichp) EddyTry
{
  NEWIMAGE::volume<float> basejac;
  NEWIMAGE::volume4D<float> grad;
  NEWIMAGE::volume<float> skrutt(pred.xsize(),pred.ysize(),pred.zsize());
  NEWIMAGE::volume<float> base = transform_model_to_scan_space(pred,scan,susc,true,skrutt,&basejac,&grad);
  ImageCoordinates basecoord = transform_coordinates_from_model_to_scan_space(pred,scan,susc,NULL,NULL);
  NEWIMAGE::volume4D<float> derivs(base.xsize(),base.ysize(),base.zsize(),scan.NDerivs(whichp));
  NEWIMAGE::copybasicproperties(scan.GetIma(),derivs);
  NEWIMAGE::volume<float> jac = pred;
  ECScan sc = scan;
  // We are relying on the order of derivatives being movement followed by EC.
  if (whichp == EDDY::ALL || whichp == EDDY::MOVEMENT) { // If we are asked for movement derivatives
    // First we calculate the movement derivatives using modulation.
    for (unsigned int i=0; i<sc.NCompoundDerivs(EDDY::MOVEMENT); i++) {
      // First calculate direct/primary derivative for the compound
      EDDY::DerivativeInstructions di = scan.GetCompoundDerivInstructions(i,EDDY::MOVEMENT);
      double p = sc.GetDerivParam(di.GetPrimaryIndex(),EDDY::MOVEMENT);
      sc.SetDerivParam(di.GetPrimaryIndex(),p+di.GetPrimaryScale(),EDDY::MOVEMENT);
      ImageCoordinates diff = transform_coordinates_from_model_to_scan_space(pred,sc,susc,NULL,&jac) - basecoord;
      derivs[di.GetPrimaryIndex()] = (diff*grad) / di.GetPrimaryScale();
      derivs[di.GetPrimaryIndex()] += base * (jac-basejac) / di.GetPrimaryScale();
      sc.SetDerivParam(di.GetPrimaryIndex(),p,EDDY::MOVEMENT);
      // Next we calculate any secondary/modulated derivatives
      if (di.IsSliceMod()) {
	for (unsigned int j=0; j<di.NSecondary(); j++) {
	  std::vector<float> smod = di.GetSliceModulator(j).GetMod();
	  for (int sl=0; sl<derivs.zsize(); sl++) {
	    for (int jj=0; jj<derivs.ysize(); jj++) {
	      for (int ii=0; ii<derivs.xsize(); ii++) {
		derivs(ii,jj,sl,di.GetSecondaryIndex(j)) = smod[sl] * derivs(ii,jj,sl,di.GetPrimaryIndex());
	      }
	    }
	  }
	}
      }
      else if (di.IsSpatiallyMod()) throw EDDY::EddyException("EddyUtils::get_partial_derivatives_in_scan_space: Spatial modulation requested");
    }
  }
  if (whichp == EDDY::ALL || whichp == EDDY::EC) { // If we are asked for EC derivatives
    // Next we calculate all the EC derivatives using direct derivatives
    unsigned int offset = (whichp == EDDY::ALL) ? scan.NDerivs(EDDY::MOVEMENT) : 0; // Hinges on MOVE before EC
    for (unsigned int i=0; i<scan.NDerivs(EDDY::EC); i++) {
      double p = sc.GetDerivParam(i,EDDY::EC);
      sc.SetDerivParam(i,p+sc.GetDerivScale(i,EDDY::EC),EDDY::EC);
      ImageCoordinates diff = transform_coordinates_from_model_to_scan_space(pred,sc,susc,NULL,&jac) - basecoord;
      derivs[offset+i] = (diff*grad) / sc.GetDerivScale(i,EDDY::EC);
      derivs[offset+i] += base * (jac-basejac) / sc.GetDerivScale(i,EDDY::EC);
      sc.SetDerivParam(i,p,EDDY::EC);
    }
  }
  return(derivs);
} EddyCatch

//
// This is the orginal version, prior to adding in slice-to-volume capability. Hopefully to be added to dead code
//
/*
NEWIMAGE::volume4D<float> EddyUtils::get_partial_derivatives_in_scan_space(// Input
									   const NEWIMAGE::volume<float>&                    pred,      // Prediction in model space
									   const EDDY::ECScan&                               scan,      // Scan space
									   std::shared_ptr<const NEWIMAGE::volume<float> >   susc,      // Susceptibility off-resonance field
									   EDDY::Parameters                                  whichp)
{
  NEWIMAGE::volume<float> basejac;
  NEWIMAGE::volume4D<float> grad;
  NEWIMAGE::volume<float> skrutt(pred.xsize(),pred.ysize(),pred.zsize());
  NEWIMAGE::volume<float> base = transform_model_to_scan_space(pred,scan,susc,true,skrutt,&basejac,&grad);
  ImageCoordinates basecoord = transform_coordinates_from_model_to_scan_space(pred,scan,susc,NULL,NULL);
  NEWIMAGE::volume4D<float> derivs(base.xsize(),base.ysize(),base.zsize(),scan.NDerivs(whichp));
  NEWIMAGE::volume<float> jac = pred;
  ECScan sc = scan;
  for (unsigned int i=0; i<sc.NDerivs(whichp); i++) {
    double p = sc.GetDerivParam(i,whichp);
    sc.SetDerivParam(i,p+sc.GetDerivScale(i,whichp),whichp);
    ImageCoordinates diff = transform_coordinates_from_model_to_scan_space(pred,sc,susc,NULL,&jac) - basecoord;
    derivs[i] = (diff*grad) / sc.GetDerivScale(i,whichp);
    derivs[i] += base * (jac-basejac) / sc.GetDerivScale(i,whichp);
    sc.SetDerivParam(i,p,whichp);
  }
  return(derivs);
}
*/

NEWIMAGE::volume4D<float> EddyUtils::get_direct_partial_derivatives_in_scan_space(// Input
										  const NEWIMAGE::volume<float>&                    pred,     // Prediction in model space
										  const EDDY::ECScan&                               scan,     // Scan space
										  std::shared_ptr<const NEWIMAGE::volume<float> >   susc,     // Susceptibility off-resonance field
										  EDDY::Parameters                                  whichp) EddyTry
{
  NEWIMAGE::volume<float> jac(pred.xsize(),pred.ysize(),pred.zsize());
  NEWIMAGE::volume<float> skrutt(pred.xsize(),pred.ysize(),pred.zsize());
  NEWIMAGE::volume<float> base = transform_model_to_scan_space(pred,scan,susc,true,skrutt,&jac,NULL);
  NEWIMAGE::volume4D<float> derivs(base.xsize(),base.ysize(),base.zsize(),scan.NDerivs(whichp));
  ECScan sc = scan;
  for (unsigned int i=0; i<sc.NDerivs(whichp); i++) {
    double p = sc.GetDerivParam(i,whichp);
    sc.SetDerivParam(i,p+sc.GetDerivScale(i,whichp),whichp);
    NEWIMAGE::volume<float> perturbed = transform_model_to_scan_space(pred,sc,susc,true,skrutt,&jac,NULL);
    derivs[i] = (perturbed-base) / sc.GetDerivScale(i,whichp);
    sc.SetDerivParam(i,p,whichp);
  }
  return(derivs);
} EddyCatch


/*
NEWIMAGE::volume<float> EddyUtils::TransformScanToModelSpace(// Input
							     const EDDY::ECScan&                               scan,
							     boost::shared_ptr<const NEWIMAGE::volume<float> > susc,
							     // Output
							     NEWIMAGE::volume<float>&                          omask)
{
  // Get total field from scan
  NEWIMAGE::volume<float> jac;
  NEWIMAGE::volume4D<float> dfield = scan.FieldForScanToModelTransform(susc,omask,jac);
  // Transform prediction using inverse RB, dfield and Jacobian
  NEWMAT::Matrix iR = scan.InverseMovementMatrix();
  NEWIMAGE::volume<float> ovol = scan.GetIma(); ovol = 0.0;
  NEWIMAGE::volume<char> mask2(ovol.xsize(),ovol.ysize(),ovol.zsize());
  NEWIMAGE::copybasicproperties(scan.GetIma(),mask2); mask2 = 1;
  NEWIMAGE::general_transform(scan.GetIma(),iR,dfield,ovol,mask2);
  // Combine all masks
  omask *= EddyUtils::ConvertMaskToFloat(mask2);
  return(jac*ovol);
}
*/

NEWIMAGE::volume<float> EddyUtils::DirectTransformScanToModelSpace(// Input
								   const EDDY::ECScan&                             scan,
								   std::shared_ptr<const NEWIMAGE::volume<float> > susc,
								   // Output
								   NEWIMAGE::volume<float>&                        omask) EddyTry
{
  NEWIMAGE::volume<float> ima = scan.GetIma();
  NEWIMAGE::volume<float> eb = scan.ECField();
  NEWIMAGE::volume4D<float> dfield = FieldUtils::Hz2VoxelDisplacements(eb,scan.GetAcqPara());
  dfield = FieldUtils::Voxel2MMDisplacements(dfield);
  NEWMAT::Matrix eye(4,4); eye=0; eye(1,1)=1.0; eye(2,2)=1.0; eye(3,3)=1.0; eye(4,4)=1.0;
  NEWIMAGE::volume<float> ovol = ima; ovol = 0.0;
  NEWIMAGE::volume<char> mask(ima.xsize(),ima.ysize(),ima.zsize());
  NEWIMAGE::apply_warp(ima,eye,dfield,eye,eye,ovol,mask);

  NEWMAT::Matrix iR = scan.InverseMovementMatrix();
  NEWIMAGE::volume<float> tmp = ovol; ovol = 0;
  NEWIMAGE::affine_transform(tmp,iR,ovol,mask);

  return(ovol);
} EddyCatch

// Right now the following function is clearly malformed. It is missing susc
/*
NEWIMAGE::volume<float> EddyUtils::DirectTransformModelToScanSpace(// Input
                                                                   const NEWIMAGE::volume<float>&                    ima,
								   const EDDY::ECScan&                               scan,
								   const EDDY::MultiBandGroups&                      mbg,
								   boost::shared_ptr<const NEWIMAGE::volume<float> > susc,
								   // Output
								   NEWIMAGE::volume<float>&                          omask)
{
  if (scan.IsSliceToVol()) {
  NEWMAT::Matrix R = scan.ForwardMovementMatrix();
  NEWIMAGE::volume<float> tmp = ima; tmp = 0;
  NEWIMAGE::volume<char> mask(tmp.xsize(),tmp.ysize(),tmp.zsize());
  NEWIMAGE::affine_transform(ima,R,tmp,mask);
  }
  else {
  }

  NEWIMAGE::volume<float> eb = scan.ECField();
  NEWIMAGE::volume4D<float> dfield = FieldUtils::Hz2VoxelDisplacements(eb,scan.GetAcqPara());
  NEWIMAGE::volume4D<float> idfield = FieldUtils::InvertDisplacementField(dfield,scan.GetAcqPara(),EddyUtils::ConvertMaskToFloat(mask),omask);
  idfield = FieldUtils::Voxel2MMDisplacements(idfield);
  NEWMAT::Matrix eye(4,4); eye=0; eye(1,1)=1.0; eye(2,2)=1.0; eye(3,3)=1.0; eye(4,4)=1.0;
  NEWIMAGE::volume<float> ovol = tmp; ovol = 0.0;
  NEWIMAGE::apply_warp(tmp,eye,idfield,eye,eye,ovol,mask);

  return(ovol);
}
*/

NEWMAT::Matrix EddyUtils::make_XtX(const NEWIMAGE::volume4D<float>& vols,
				   const NEWIMAGE::volume<float>&   mask) EddyTry
{
  NEWMAT::Matrix XtX(vols.tsize(),vols.tsize());
  XtX = 0.0;
  for (int r=1; r<=vols.tsize(); r++) {
    for (int c=r; c<=vols.tsize(); c++) {
      for (NEWIMAGE::volume<float>::fast_const_iterator rit=vols.fbegin(r-1), ritend=vols.fend(r-1), cit=vols.fbegin(c-1), mit=mask.fbegin(); rit!=ritend; ++rit, ++cit, ++mit) {
	if (*mit) XtX(r,c) += (*rit)*(*cit);
      }
    }
  }
  for (int r=2; r<=vols.tsize(); r++) {
    for (int c=1; c<r; c++) XtX(r,c) = XtX(c,r);
  }
  return(XtX);
} EddyCatch

NEWMAT::ColumnVector EddyUtils::make_Xty(const NEWIMAGE::volume4D<float>& Xvols,
					 const NEWIMAGE::volume<float>&   Yvol,
					 const NEWIMAGE::volume<float>&   mask) EddyTry
{
  NEWMAT::ColumnVector Xty(Xvols.tsize());
  Xty = 0.0;
  for (int r=1; r<=Xvols.tsize(); r++) {
    for (NEWIMAGE::volume<float>::fast_const_iterator Xit=Xvols.fbegin(r-1), Xend=Xvols.fend(r-1), Yit=Yvol.fbegin(), mit=mask.fbegin(); Xit!=Xend; ++Xit, ++Yit, ++mit) {
      if (*mit) Xty(r) += (*Xit)*(*Yit);
    }
  }
  return(Xty);
} EddyCatch

// Has been modified for slice-to-vol

void EddyUtils::transform_coordinates(// Input
				      const NEWIMAGE::volume<float>&    f,
				      const NEWIMAGE::volume4D<float>&  d,
				      const NEWMAT::Matrix&             M,
				      std::vector<unsigned int>         slices,
				      // Input/Output
				      ImageCoordinates&                 c,
                                      // Output
				      NEWIMAGE::volume<float>           *omask) EddyTry
{
  NEWMAT::Matrix iA = d[0].sampling_mat();

  float A11=iA(1,1), A12=iA(1,2), A13=iA(1,3), A14=iA(1,4);
  float A21=iA(2,1), A22=iA(2,2), A23=iA(2,3), A24=iA(2,4);
  float A31=iA(3,1), A32=iA(3,2), A33=iA(3,3), A34=iA(3,4);

  // Create a matrix mapping from mm-coordinates in volume i
  // to voxel coordinates in volume f. If the matrix M is empty
  // this is simply a mm->voxel mapping for volume f

  NEWMAT::Matrix iM = f.sampling_mat().i() * M.i();

  float M11=iM(1,1), M12=iM(1,2), M13=iM(1,3), M14=iM(1,4);
  float M21=iM(2,1), M22=iM(2,2), M23=iM(2,3), M24=iM(2,4);
  float M31=iM(3,1), M32=iM(3,2), M33=iM(3,3), M34=iM(3,4);

  // If no slices were specified, do all slices
  if (slices.size() == 0) { slices.resize(c.NZ()); for (unsigned int z=0; z<c.NZ(); z++) slices[z] = z; }
  else if (slices.size() > c.NZ()) throw EddyException("EddyUtils::transform_coordinates: slices vector too long");
  else { for (unsigned int z=0; z<slices.size(); z++) if (slices[z] >= c.NZ()) throw EddyException("EddyUtils::transform_coordinates: slices vector has invalid entry");}

  for (unsigned int k=0; k<slices.size(); k++) {
    unsigned int z = slices[k];
    unsigned int index = z * c.NY() * c.NX();
    float xtmp1 = A13*z + A14;
    float ytmp1 = A23*z + A24;
    float ztmp1 = A33*z + A34;
    for (unsigned int y=0; y<c.NY(); y++) {
      float xtmp2 = xtmp1 + A12*y;
      float ytmp2 = ytmp1 + A22*y;
      float ztmp2 = ztmp1 + A32*y;
      for (unsigned int x=0; x<c.NX(); x++) {
	float o1 = xtmp2 + A11*x + d(x,y,z,0);
	float o2 = ytmp2 + A21*x + d(x,y,z,1);
	float o3 = ztmp2 + A31*x + d(x,y,z,2);
	if (omask) (*omask)(x,y,z) = 1;  // So far, so good
	c.x(index) = M11*o1 + M12*o2 + M13*o3 + M14;
	c.y(index) = M21*o1 + M22*o2 + M23*o3 + M24;
	c.z(index) = M31*o1 + M32*o2 + M33*o3 + M34;
	if (omask) (*omask)(x,y,z) *= (f.valid(c.x(index),c.y(index),c.z(index))) ? 1 : 0; // Kosher only if valid in both d and s
        index++;
      }
    }
  }
  return;
} EddyCatch

bool EddyUtils::UpdateMakesSense(const EDDY::ECScan&           scan,
				 const NEWMAT::ColumnVector&   update) EddyTry
{
  double maxtr = 5.0;
  double maxpetr = 10.0;
  double maxrot = 3.1416*5.0/180.0;
  double maxfirst = 0.27; // 6 times standard deviation of empirical updates on first iteration
  double maxsecond = 0.01; // 6 times standard deviation of empirical updates on first iteration
  NEWMAT::ColumnVector pevec = scan.GetAcqPara().PhaseEncodeVector();
  unsigned int morder = scan.GetMovementModelOrder() + 1;

  // Translations
  if (pevec(1) != 0) {
    for (unsigned int i=0; i<morder; i++) if (std::abs(update(i+1)) > maxpetr) return(false);
    for (unsigned int i=morder; i<3*morder; i++) if (std::abs(update(i+1)) > maxtr) return(false);
  }
  else {
    for (unsigned int i=0; i<morder; i++) if (std::abs(update(i+1)) > maxtr) return(false);
    for (unsigned int i=morder; i<2*morder; i++) if (std::abs(update(i+1)) > maxpetr) return(false);
    for (unsigned int i=2*morder; i<3*morder; i++) if (std::abs(update(i+1)) > maxtr) return(false);
  }
  // Rotations
  for (unsigned int i=3*morder; i<6*morder; i++) {
    if (std::abs(update(i+1)) > maxrot) return(false);
  }
  // First order EC
  if (scan.Model() == Linear || scan.Model() == Quadratic || scan.Model() == Cubic) {
    for (unsigned int i=6*morder; i<6*morder+3; i++) {
      if (std::abs(update(i+1)) > maxfirst) return(false);
    }
  }
  // Second order EC
  if (scan.Model() == Quadratic || scan.Model() == Cubic) {
    for (unsigned int i=6*morder+3; i<6*morder+9; i++) {
      if (std::abs(update(i+1)) > maxsecond) return(false);
    }
  }
  // I currently lack data for 3rd order EC
  return(true);
} EddyCatch

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class FieldUtils
//
// Helper Class used to perform various useful calculations
// on displacement fields.
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

NEWIMAGE::volume4D<float> FieldUtils::Hz2VoxelDisplacements(const NEWIMAGE::volume<float>& hzfield,
                                                            const AcqPara&                 acqp) EddyTry
{
  NEWIMAGE::volume4D<float> dfield(hzfield.xsize(),hzfield.ysize(),hzfield.zsize(),3);
  NEWIMAGE::copybasicproperties(hzfield,dfield);
  for (int i=0; i<3; i++) dfield[i] = float((acqp.PhaseEncodeVector())(i+1) * acqp.ReadOutTime()) * hzfield;
  return(dfield);
} EddyCatch

NEWIMAGE::volume4D<float> FieldUtils::Hz2MMDisplacements(const NEWIMAGE::volume<float>& hzfield,
                                                         const AcqPara&                 acqp) EddyTry
{
  NEWIMAGE::volume4D<float> dfield(hzfield.xsize(),hzfield.ysize(),hzfield.zsize(),3);
  NEWIMAGE::copybasicproperties(hzfield,dfield);
  dfield[0] = float(hzfield.xdim()*(acqp.PhaseEncodeVector())(1) * acqp.ReadOutTime()) * hzfield;
  dfield[1] = float(hzfield.ydim()*(acqp.PhaseEncodeVector())(2) * acqp.ReadOutTime()) * hzfield;
  dfield[2] = float(hzfield.zdim()*(acqp.PhaseEncodeVector())(3) * acqp.ReadOutTime()) * hzfield;
  return(dfield);
} EddyCatch

/////////////////////////////////////////////////////////////////////
//
// Inverts a 1D displacementfield. The input field should be in units
// of voxels and the output will be too.
//
/////////////////////////////////////////////////////////////////////
NEWIMAGE::volume<float> FieldUtils::Invert1DDisplacementField(// Input
							      const NEWIMAGE::volume<float>& dfield,
							      const AcqPara&                 acqp,
							      const NEWIMAGE::volume<float>& inmask,
							      // Output
							      NEWIMAGE::volume<float>&       omask) EddyTry
{
  NEWIMAGE::volume<float> fc = dfield;   // fc : field copy
  NEWIMAGE::volume<float> imc = inmask;  // imc: inmask copy
  // Make it so that we invert in first (x) direction
  unsigned int d=0;
  for (; d<3; d++) if ((acqp.PhaseEncodeVector())(d+1)) break;
  if (d==1) {
    fc.swapdimensions(2,1,3);
    imc.swapdimensions(2,1,3);
    omask.swapdimensions(2,1,3);
  }
  else if (d==2) {
    fc.swapdimensions(3,2,1);
    imc.swapdimensions(3,2,1);
    omask.swapdimensions(3,2,1);
  }
  NEWIMAGE::volume<float> idf = fc;    // idf : inverse displacement field
  // Do the inversion
  for (int k=0; k<idf.zsize(); k++) {
    for (int j=0; j<idf.ysize(); j++) {
      int oi=0;
      for (int i=0; i<idf.xsize(); i++) {
	int ii=oi;
	for (; ii<idf.xsize() && fc(ii,j,k)+ii<i; ii++) ; // On purpose
	if (ii>0 && ii<idf.xsize()) { // If we are in valid range
	  idf(i,j,k) = ii - i - 1.0 + float(i+1-ii-fc(ii-1,j,k))/float(fc(ii,j,k)+1.0-fc(ii-1,j,k));
          if (imc(ii-1,j,k)) omask(i,j,k) = 1.0;
	  else omask(i,j,k) = 0.0;
	}
	else {
	  idf(i,j,k) = FLT_MAX;    // Tag for further processing
	  omask(i,j,k) = 0.0;
	}
	oi = std::max(0,ii-1);
      }
      // Process NaN's at beginning of column
      int ii=0;
      for (ii=0; ii<idf.xsize()-1 && idf(ii,j,k)==FLT_MAX; ii++) ; // On purpose
      for (; ii>0; ii--) idf(ii-1,j,k) = idf(ii,j,k);
      // Process NaN's at end of column
      for (ii=idf.xsize()-1; ii>0 && idf(ii,j,k)==FLT_MAX; ii--) ; // On purpose
      for (; ii<idf.xsize()-1; ii++) idf(ii+1,j,k) = idf(ii,j,k);
    }
  }
  // Swap back to original orientation
  if (d==1) {
    idf.swapdimensions(2,1,3);
    omask.swapdimensions(2,1,3);
  }
  else if (d==2) {
    idf.swapdimensions(3,2,1);
    omask.swapdimensions(3,2,1);
  }

  return(idf);
} EddyCatch

/////////////////////////////////////////////////////////////////////
//
// Inverts a "3D" displacementfield. The input field should be in units
// of voxels and the output will be too. The current implementation
// expects displacements to be in one direction only (i.e. 1D).
//
/////////////////////////////////////////////////////////////////////
NEWIMAGE::volume4D<float> FieldUtils::Invert3DDisplacementField(// Input
								const NEWIMAGE::volume4D<float>& dfield,
								const AcqPara&                   acqp,
								const NEWIMAGE::volume<float>&   inmask,
								// Output
								NEWIMAGE::volume<float>&         omask) EddyTry
{
  NEWIMAGE::volume4D<float> idfield = dfield;
  idfield = 0.0;
  unsigned int cnt=0;
  for (unsigned int i=0; i<3; i++) if ((acqp.PhaseEncodeVector())(i+1)) cnt++;
  if (cnt != 1) throw EddyException("FieldUtils::InvertDisplacementField: Phase encode vector must have exactly one non-zero component");
  unsigned int i=0;
  for (; i<3; i++) if ((acqp.PhaseEncodeVector())(i+1)) break;
  idfield[i] = Invert1DDisplacementField(dfield[i],acqp,inmask,omask);

  return(idfield);
} EddyCatch

/////////////////////////////////////////////////////////////////////
//
// Calculates the Jacobian determinant of a 3D displacement field.
// The field must be in units of voxels and in the present
// implementation it must also be inherently 1D.
//
/////////////////////////////////////////////////////////////////////
NEWIMAGE::volume<float> FieldUtils::GetJacobian(const NEWIMAGE::volume4D<float>& dfield,
                                                const AcqPara&                   acqp) EddyTry
{
  unsigned int cnt=0;
  for (unsigned int i=0; i<3; i++) if ((acqp.PhaseEncodeVector())(i+1)) cnt++;
  if (cnt != 1) throw EddyException("FieldUtils::GetJacobian: Phase encode vector must have exactly one non-zero component");
  unsigned int i=0;
  for (; i<3; i++) if ((acqp.PhaseEncodeVector())(i+1)) break;

  NEWIMAGE::volume<float> jacfield = GetJacobianFrom1DField(dfield[i],i);

  return(jacfield);
} EddyCatch

/////////////////////////////////////////////////////////////////////
//
// Calculates the Jacobian determinant of a 1D displacement field.
// The field must be in units of voxels.
//
/////////////////////////////////////////////////////////////////////
NEWIMAGE::volume<float> FieldUtils::GetJacobianFrom1DField(const NEWIMAGE::volume<float>& dfield,
                                                           unsigned int                   dir) EddyTry
{
  // Calculate spline coefficients for displacement field
  std::vector<unsigned int>                        dim(3,0);
  dim[0] = dfield.xsize(); dim[1] = dfield.ysize(); dim[2] = dfield.zsize();
  std::vector<SPLINTERPOLATOR::ExtrapolationType>  ep(3,SPLINTERPOLATOR::Mirror);
  SPLINTERPOLATOR::Splinterpolator<float> spc(dfield.fbegin(),dim,ep,3,false);
  // Get Jacobian at voxel centres
  NEWIMAGE::volume<float> jacf = dfield;
  for (int k=0; k<dfield.zsize(); k++) {
    for (int j=0; j<dfield.ysize(); j++) {
      for (int i=0; i<dfield.xsize(); i++) {
        jacf(i,j,k) = 1.0 + spc.DerivXYZ(i,j,k,dir);
      }
    }
  }
  return(jacf);
} EddyCatch

/****************************************************************//**
*
* \brief Performs common construction tasks for s2vQuant
*
*
********************************************************************/
void s2vQuant::common_construction() EddyTry
{
  if (!_sm.Scan(0,ANY).IsSliceToVol()) throw EddyException("s2vQuant::common_construction: Data is not slice-to-vol");;

  std::vector<unsigned int> icsl;
  if (_sm.MultiBand().MBFactor() == 1) icsl = _sm.IntraCerebralSlices(500); // N.B. Hardcoded. Might want to move up
  _tr.ReSize(3,_sm.NScans(ANY));
  _rot.ReSize(3,_sm.NScans(ANY));
  for (unsigned int i=0; i<_sm.NScans(ANY); i++) {
    for (unsigned int j=0; j<3; j++) {
      _tr(j+1,i+1) = _sm.Scan(i,ANY).GetMovementStd(j,icsl);
      _rot(j+1,i+1) = 180.0 * _sm.Scan(i,ANY).GetMovementStd(3+j,icsl) / 3.141592653589793;
    }
  }
} EddyCatch

/****************************************************************//**
*
* \brief Returns a vector of indices to volumes with little movement
*
*
*
********************************************************************/
std::vector<unsigned int> s2vQuant::FindStillVolumes(ScanType                         st,
						     const std::vector<unsigned int>& mbsp) const EddyTry
{
  std::vector<unsigned int> rval;
  for (unsigned int i=0; i<_sm.NScans(st); i++) {
    unsigned int j = i;
    if (st==B0) j = _sm.Getb02GlobalIndexMapping(i);
    else if (st==DWI) j = _sm.GetDwi2GlobalIndexMapping(i);
    bool is_still = true;
    for (unsigned int pi=0; pi<mbsp.size(); pi++) {
      if (mbsp[pi] < 3) {
	NEWMAT::ColumnVector tmp = _tr.Column(j+1);
	if (tmp(mbsp[pi]+1) > _trth) is_still = false;
      }
      else if (mbsp[pi] > 2 && mbsp[pi] < 6) {
	NEWMAT::ColumnVector tmp = _rot.Column(j+1);
	if (tmp(mbsp[pi]+1-3) > _rotth) is_still = false;
      }
      else throw EddyException("s2vQuant::FindStillVolumes: mbsp out of range");
    }
    if (is_still) rval.push_back(i);
  }
  return(rval);
} EddyCatch
