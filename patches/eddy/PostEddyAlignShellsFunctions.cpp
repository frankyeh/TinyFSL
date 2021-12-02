/*! \file PostEddyAlignShellsFunctions.cpp
    \brief Contains some high level global functions used to
    align shells to each other and to b0 after the "main eddy"
    has completed.
*/

#include <iostream>

#include "topup/topup_file_io.h"
#include "EddyCommandLineOptions.h"
#include "PostEddyCF.h"
#include "PostEddyAlignShellsFunctions.h"
#ifdef COMPILE_GPU
#include "cuda/EddyGpuUtils.h"
#endif

using namespace std;


namespace EDDY {

void PEASUtils::PostEddyAlignShells(// Input
				    const EddyCommandLineOptions&   clo,
				    bool                            upe, // Update parameter estimates or not
				    // Input/Output
				    ECScanManager&                  sm) EddyTry
{
  std::vector<double>                               grpb;
  std::vector<NEWMAT::ColumnVector>                 mi_mov_par;
  std::vector<NEWMAT::ColumnVector>                 mi_mp_for_updates;
  std::vector<std::vector<NEWMAT::ColumnVector> >   mi_cmov_par;
  std::vector<NEWMAT::ColumnVector>                 b0_mov_par;
  std::vector<NEWMAT::ColumnVector>                 b0_mp_for_updates;

  if (sm.B0sAreUsefulForPEAS()) {
    if (clo.VeryVerbose()) cout << "Using interspersed b0's to estimate between shell movement" << endl;
    PEASUtils::align_shells_using_interspersed_B0_scans(clo,sm,grpb,b0_mov_par,b0_mp_for_updates);
  }
  if (clo.VeryVerbose()) cout << "Using MI to estimate between shell movement" << endl;
  PEASUtils::align_shells_using_MI(clo,false,sm,grpb,mi_mov_par,mi_cmov_par,mi_mp_for_updates); // Align based on MI between shell means

  // Write report
  PEASUtils::write_post_eddy_align_shells_report(mi_mov_par,mi_mp_for_updates,mi_cmov_par,b0_mov_par,b0_mp_for_updates,grpb,upe,clo);

  // Update parameter estimates if requested
  if (upe) {
    std::vector<NEWMAT::ColumnVector> mp_for_updates;
    if (clo.UseB0sToAlignShellsPostEddy()) mp_for_updates = b0_mp_for_updates;
    else mp_for_updates = mi_mp_for_updates;

    std::vector<std::vector<unsigned int> > dwi_indx = sm.GetShellIndicies(grpb);

    for (unsigned int i=0; i<mp_for_updates.size(); i++) {
      if (clo.VeryVerbose()) cout << "Parameters for shell " << grpb[i] << " to b0" << endl;
      if (clo.VeryVerbose()) cout << "mp_for_updates = " << mp_for_updates[i] << endl;
    }
    if (clo.VeryVerbose()) cout << "Updating parameter estimates" << endl;
    for (unsigned int i=0; i<dwi_indx.size(); i++) {
      if (clo.VeryVerbose()) cout << "Updating parmeter estimates for shell " << i << endl;
      PEASUtils::update_mov_par_estimates(mp_for_updates[i],dwi_indx[i],sm);
    }
  }
  return;
} EddyCatch

void PEASUtils::PostEddyAlignShellsAlongPE(// Input
					   const EddyCommandLineOptions&   clo,
					   bool                            upe, // Update parameter estimates or not
					   // Input/Output
					   ECScanManager&                  sm) EddyTry
{
  // If there are scans with PE in x AND y there should be no ambiguity
  if (sm.HasPEinXandY()) return;

  std::vector<double>                               grpb;
  std::vector<NEWMAT::ColumnVector>                 mi_mov_par;
  std::vector<NEWMAT::ColumnVector>                 mi_mp_for_updates;
  std::vector<std::vector<NEWMAT::ColumnVector> >   mi_cmov_par;

  if (clo.VeryVerbose()) cout << "Using MI to estimate between shell translation along PE" << endl;
  PEASUtils::align_shells_using_MI(clo,true,sm,grpb,mi_mov_par,mi_cmov_par,mi_mp_for_updates); // Align based on MI between shell means

  // Write report
  PEASUtils::write_post_eddy_align_shells_along_PE_report(mi_mov_par,mi_mp_for_updates,mi_cmov_par,grpb,upe,clo);

  // Update parameter estimates if requested
  if (upe) {
    std::vector<std::vector<unsigned int> > dwi_indx = sm.GetShellIndicies(grpb);

    for (unsigned int i=0; i<mi_mp_for_updates.size(); i++) {
      if (clo.VeryVerbose()) cout << "Parameters for shell " << grpb[i] << " to b0" << endl;
      if (clo.VeryVerbose()) cout << "mp_for_updates = " << mi_mp_for_updates[i] << endl;
    }
    if (clo.VeryVerbose()) cout << "Updating parameter estimates" << endl;
    for (unsigned int i=0; i<dwi_indx.size(); i++) {
      if (clo.VeryVerbose()) cout << "Updating parmeter estimates for shell " << i << endl;
      PEASUtils::update_mov_par_estimates(mi_mp_for_updates[i],dwi_indx[i],sm);
    }
  }
  return;
} EddyCatch

void PEASUtils::WritePostEddyBetweenShellMIValues(// Input
						  const EddyCommandLineOptions&     clo,
						  const ECScanManager&              sm,
						  const std::vector<unsigned int>&  n,
						  const std::vector<double>&        first,
						  const std::vector<double>&        last,
						  const std::string&                bfname) EddyTry
{
  // Get mean b0 volume
  NEWIMAGE::volume<float>     b0_mask = sm.Scan(0,ANY).GetIma(); b0_mask=1.0;
  std::vector<unsigned int>   b0_indx = sm.GetB0Indicies();
  NEWIMAGE::volume<float>     b0_mean = PEASUtils::get_mean_scan(clo,sm,b0_indx,b0_mask);
  // NEWIMAGE::write_volume(b0_mean,"mean_b0_volume");

  // Get mean volumes for all shells
  if (clo.VeryVerbose()) cout << "Calculating shell means" << endl;
  std::vector<NEWIMAGE::volume<float> >  dwi_means;
  std::vector<double>                    grpb;
  NEWIMAGE::volume<float>                mask = sm.Scan(0,ANY).GetIma(); mask=1.0;
  if (sm.IsShelled()) { // Get shell means
    std::vector<std::vector<unsigned int> > dwi_indx = sm.GetShellIndicies(grpb);
    dwi_means.resize(dwi_indx.size());
    NEWIMAGE::volume<float> tmpmask;
    for (unsigned int i=0; i<dwi_indx.size(); i++) {
      tmpmask = 1.0;
      dwi_means[i] = PEASUtils::get_mean_scan(clo,sm,dwi_indx[i],tmpmask);
      mask *= tmpmask;
    }
  }
  else { // Get mean of all dwi's
  }
  mask *= b0_mask; // Mask valid for b0 and dwi

  // Get and write MI values for all combinations of b0 and dwi shells
  for (unsigned int i=0; i<dwi_means.size(); i++) {
    char ofname[256]; sprintf(ofname,"%s_b0_b%d",bfname.c_str(),100*static_cast<int>(std::round(grpb[i]/100.0)));
    write_between_shell_MI_values(b0_mean,dwi_means[i],mask,ofname,n,first,last);
  }
} EddyCatch

/****************************************************************//**
*
*  A static and private function in PEASUtils
*  \param[in] sm Collection of all scans.
*  \param[out] means a vector of mean-volumes (one per shell)
*  is present for all input scans in sm.
*  \return The number of shells/means
*
********************************************************************/
NEWIMAGE::volume<float> PEASUtils::get_mean_scan(// Input
						 const EddyCommandLineOptions&     clo,
						 const ECScanManager&              sm,
						 const std::vector<unsigned int>&  indx,
						 // Output
						 NEWIMAGE::volume<float>&          mask) EddyTry
{
  NEWIMAGE::volume<float> mean = sm.Scan(0,ANY).GetIma(); mean = 0.0;
  mask = sm.Scan(0,ANY).GetIma(); EddyUtils::SetTrilinearInterp(mask); mask = 1.0;
  if (clo.VeryVerbose()) cout << "Calculating shell means";
  if (clo.VeryVerbose()) cout << endl << "Scan: " << endl;
#ifdef COMPILE_GPU
  NEWIMAGE::volume<float> tmpmask = sm.Scan(0,ANY).GetIma();
  EddyUtils::SetTrilinearInterp(tmpmask);
  for (unsigned int i=0; i<indx.size(); i++) {
    if (clo.VeryVerbose()) printf("%d ",indx[i]);
    tmpmask = 1.0;
    mean += EddyGpuUtils::GetUnwarpedScan(sm.Scan(indx[i],ANY),sm.GetSuscHzOffResField(indx[i],ANY),sm.GetBiasField(),false,&tmpmask);
    mask *= tmpmask;
  }
#else
#pragma omp parallel for shared(sm,mean,mask)
  for (int i=0; i<int(indx.size()); i++) {
    if (clo.VeryVerbose()) { cout << indx[i] << " "; cout.flush(); }
    NEWIMAGE::volume<float> tmpmask = sm.Scan(0,ANY).GetIma();
    EddyUtils::SetTrilinearInterp(tmpmask); tmpmask = 1.0;
    NEWIMAGE::volume<float> tmpima = sm.GetUnwarpedScan(indx[i],tmpmask,ANY);
#pragma omp critical
    {
      mean += tmpima;
      mask *= tmpmask;
    }
  }
#endif
  if (clo.VeryVerbose()) printf("\n");
  mean /= indx.size();

  return(mean);
} EddyCatch

NEWMAT::ColumnVector PEASUtils::register_volumes(// Input
						 const NEWIMAGE::volume<float>& ref,
						 const NEWIMAGE::volume<float>& ima,
						 const NEWIMAGE::volume<float>& mask,
						 // Output
						 NEWIMAGE::volume<float>&       rima) EddyTry
{
  EddyUtils::SetSplineInterp(ima);
  EddyUtils::SetTrilinearInterp(mask);
  MISCMATHS::NonlinParam nlpar(6,MISCMATHS::NL_NM);
  // Set starting simplex to 1mm and 1 degree.
  NEWMAT::ColumnVector ss(6); ss=1.0; for (int i=3; i<6; i++) ss(i+1) = 3.1415 / 180.0;
  nlpar.SetStartAmoeba(ss);
  // Make MI cost-function object with 256 bins
  // 256 is MJ's recommendation, and also looks the smoothest in my testing.
  PostEddyCF ecf(ref,ima,mask,256);
  // Run optimisation
  MISCMATHS::nonlin(nlpar,ecf);
  rima = ecf.GetTransformedIma(nlpar.Par());

  return(nlpar.Par());
} EddyCatch

NEWMAT::ColumnVector PEASUtils::register_volumes_along_PE(// Input
							  const NEWIMAGE::volume<float>& ref,
							  const NEWIMAGE::volume<float>& ima,
							  const NEWIMAGE::volume<float>& mask,
							  unsigned int                   pe_dir,
							  // Output
							  NEWIMAGE::volume<float>&       rima) EddyTry
{
  EddyUtils::SetSplineInterp(ima);
  EddyUtils::SetTrilinearInterp(mask);
  MISCMATHS::NonlinParam nlpar(1,MISCMATHS::NL_NM);
  // Set starting simplex (actually just a line in this case) to 1mm
  NEWMAT::ColumnVector ss(1); ss(1)=1.0;
  nlpar.SetStartAmoeba(ss);
  // Make MI cost-function object with 256 bins
  // 256 is MJ's recommendation, and also looks the smoothest in my testing.
  PostEddyCF ecf(ref,ima,mask,256,pe_dir);
  // Run optimisation
  MISCMATHS::nonlin(nlpar,ecf);
  rima = ecf.GetTransformedIma(nlpar.Par());

  return(nlpar.Par());
} EddyCatch

std::vector<NEWMAT::ColumnVector> PEASUtils::collate_mov_par_estimates_for_use(const std::vector<NEWMAT::ColumnVector>&                mp,
									       const std::vector<std::vector<NEWMAT::ColumnVector> >&  cmp,
									       const NEWIMAGE::volume<float>&                          ima) EddyTry
{
  /*
  // This bit of code combines estimates of lowest b-val shell to
  // b0 with estimates from all other shells to lowest b-val
  std::vector<NEWMAT::ColumnVector> omp = mp;
  for (unsigned int i=1; i<omp.size(); i++) {
    NEWMAT::Matrix A = TOPUP::MovePar2Matrix(mp[0],ima) * TOPUP::MovePar2Matrix(cmp[0][i],ima);
    omp[i] = TOPUP::Matrix2MovePar(A,ima);
  }
  */

  // This bit of code uses the direct estimates of each shell to b0
  std::vector<NEWMAT::ColumnVector> omp = mp;

  /*
  // This bit of code uses the direct estimates of the lowest shell to b0
  // and then assumes that the shells are already aligned to each other.
  std::vector<NEWMAT::ColumnVector> omp = mp;
  for (unsigned int i=1; i<omp.size(); i++) omp[i] = omp[0];
  */

  return(omp);
} EddyCatch

void PEASUtils::write_post_eddy_align_shells_report(const std::vector<NEWMAT::ColumnVector>&                mi_dmp,
						    const std::vector<NEWMAT::ColumnVector>&                mi_ump,
						    const std::vector<std::vector<NEWMAT::ColumnVector> >&  mi_cmp,
						    const std::vector<NEWMAT::ColumnVector>&                b0_dmp,
						    const std::vector<NEWMAT::ColumnVector>&                b0_ump,
						    const std::vector<double>&                              grpb,
						    bool                                                    upe,
						    const EddyCommandLineOptions&                           clo) EddyTry
{
  try {
    std::ofstream  file;
    double pi = 3.14159;
    file.open(clo.PeasReportFname().c_str(),ios::out|ios::trunc);
    file << "These between shell parameters were calculated using MI between shell means" << endl;
    file << setprecision(3) << "Movement parameters relative b0-shell from direct registration to mean b0" << endl;
    for (unsigned int i=0; i<mi_dmp.size(); i++) {
      file << "Shell " << grpb[i] << " to b0-shell" << endl;
      file << setw(10) << "x-tr (mm)" << setw(10) << "y-tr (mm)" << setw(10) << "z-tr (mm)" << setw(12) << "x-rot (deg)" << setw(12) << "y-rot (deg)" << setw(12) << "z-rot (deg)" << endl;
      file << setw(10) << mi_dmp[i](1) << setw(10) << mi_dmp[i](2) << setw(10) << mi_dmp[i](3) << setw(12) << mi_dmp[i](4)*180.0/pi << setw(12) << mi_dmp[i](5)*180.0/pi << setw(12) << mi_dmp[i](6)*180.0/pi << endl;
    }
    file << endl << "Relative movement parameters between the shells" << endl;
    for (unsigned int i=1; i<mi_cmp[0].size(); i++) {
      file << "Shell " << grpb[i] << " to shell " << grpb[0] << endl;
      file << setw(10) << "x-tr (mm)" << setw(10) << "y-tr (mm)" << setw(10) << "z-tr (mm)" << setw(12) << "x-rot (deg)" << setw(12) << "y-rot (deg)" << setw(12) << "z-rot (deg)" << endl;
      file << setw(10) << mi_cmp[0][i](1) << setw(10) << mi_cmp[0][i](2) << setw(10) << mi_cmp[0][i](3);
      file << setw(12) << mi_cmp[0][i](4)*180.0/pi << setw(12) << mi_cmp[0][i](5)*180.0/pi << setw(12) << mi_cmp[0][i](6)*180.0/pi << endl;
    }
    file << endl << "Deduced movement parameters relative b0-shell" << endl;
    for (unsigned int i=0; i<mi_ump.size(); i++) {
      file << "Shell " << grpb[i] << " to b0-shell" << endl;
      file << setw(10) << "x-tr (mm)" << setw(10) << "y-tr (mm)" << setw(10) << "z-tr (mm)" << setw(12) << "x-rot (deg)" << setw(12) << "y-rot (deg)" << setw(12) << "z-rot (deg)" << endl;
      file << setw(10) << mi_ump[i](1) << setw(10) << mi_ump[i](2) << setw(10) << mi_ump[i](3) << setw(12) << mi_ump[i](4)*180.0/pi << setw(12) << mi_ump[i](5)*180.0/pi << setw(12) << mi_ump[i](6)*180.0/pi << endl;
    }
    file << endl << endl;
    if (b0_dmp.size()) { // If parameters were also estimated using interspersed b0s
      file << "These between shell parameters were calculated using interspersed b0-volumes" << endl;
      file << setprecision(3) << "Movement parameters relative b0-shell" << endl;
      for (unsigned int i=0; i<b0_dmp.size(); i++) {
	file << "Shell " << grpb[i] << " to b0-shell" << endl;
	file << setw(10) << "x-tr (mm)" << setw(10) << "y-tr (mm)" << setw(10) << "z-tr (mm)" << setw(12) << "x-rot (deg)" << setw(12) << "y-rot (deg)" << setw(12) << "z-rot (deg)" << endl;
	file << setw(10) << b0_dmp[i](1) << setw(10) << b0_dmp[i](2) << setw(10) << b0_dmp[i](3) << setw(12) << b0_dmp[i](4)*180.0/pi << setw(12) << b0_dmp[i](5)*180.0/pi << setw(12) << b0_dmp[i](6)*180.0/pi << endl;
      }
      file << endl << endl;
    }
    if (!upe) file << "The movement parameters presented above have been calculated but not used";
    else {
      file << "These are the movement parameters that have been applied to the data" << endl << endl;
      std::vector<NEWMAT::ColumnVector> ump;
      if (clo.UseB0sToAlignShellsPostEddy()) ump = b0_ump;
      else ump = mi_ump;
      for (unsigned int i=0; i<ump.size(); i++) {
	file << "Shell " << grpb[i] << " to b0-shell" << endl;
	file << setw(10) << "x-tr (mm)" << setw(10) << "y-tr (mm)" << setw(10) << "z-tr (mm)" << setw(12) << "x-rot (deg)" << setw(12) << "y-rot (deg)" << setw(12) << "z-rot (deg)" << endl;
	file << setw(10) << ump[i](1) << setw(10) << ump[i](2) << setw(10) << ump[i](3) << setw(12) << ump[i](4)*180.0/pi << setw(12) << ump[i](5)*180.0/pi << setw(12) << ump[i](6)*180.0/pi << endl;
      }
    }
    file.close();
  }
  catch (...) {
    throw EddyException("EDDY::PEASUtils::write_post_eddy_align_shells_report: Failed writing file.");
  }
} EddyCatch

void PEASUtils::write_post_eddy_align_shells_along_PE_report(const std::vector<NEWMAT::ColumnVector>&                mi_dmp,
							     const std::vector<NEWMAT::ColumnVector>&                mi_ump,
							     const std::vector<std::vector<NEWMAT::ColumnVector> >&  mi_cmp,
							     const std::vector<double>&                              grpb,
							     bool                                                    upe,
							     const EddyCommandLineOptions&                           clo) EddyTry
{
  try {
    std::ofstream  file;
    file.open(clo.PeasAlongPEReportFname().c_str(),ios::out|ios::trunc);
    file << "These between shell PE-translations were calculated using MI between shell means" << endl;
    file << setprecision(3) << "PE-translations (mm) relative b0-shell from direct registration to mean b0" << endl;
    for (unsigned int i=0; i<mi_dmp.size(); i++) {
      file << "Shell " << grpb[i] << " to b0-shell: PE-translation = " << mi_dmp[i](1) << " mm" << endl;
    }
    file << endl << "Relative PE-translations (mm) between the shells" << endl;
    for (unsigned int i=1; i<mi_cmp[0].size(); i++) {
      file << "Shell " << grpb[i] << " to shell " << grpb[0] << ": PE-translation = " << mi_cmp[0][i](1) << " mm" << endl;
    }
    file << endl << "Deduced PE-translations (mm) relative b0-shell" << endl;
    for (unsigned int i=0; i<mi_ump.size(); i++) {
      file << "Shell " << grpb[i] << " to b0-shell: PE-translation = " << mi_ump[i](1) << " mm" << endl;
    }
    file << endl << endl;
    if (!upe) file << "The PE-translations presented above have been calculated but not used";
    else {
      file << "These are the PE-translations (mm) that have been applied to the data" << endl << endl;
      for (unsigned int i=0; i<mi_ump.size(); i++) {
	file << "Shell " << grpb[i] << " to b0-shell: PE-translation = " << mi_ump[i](1) << " mm" << endl;
      }
    }
    file.close();
  }
  catch (...) {
    throw EddyException("EDDY::WritePostEddyAlignShellsReport: Failed writing file.");
  }
} EddyCatch

void PEASUtils::update_mov_par_estimates(// Input
					 const NEWMAT::ColumnVector&       mp,
					 const std::vector<unsigned int>&  indx,
					 // Input/output
					 ECScanManager&                    sm) EddyTry
{
  NEWMAT::Matrix A;
  if (mp.Nrows()==1) {
    NEWMAT::ColumnVector tmp(6); tmp = 0.0;
    if (sm.HasPEinX()) tmp(1) = mp(1); else tmp(2) = mp(1);
    A = TOPUP::MovePar2Matrix(tmp,sm.Scan(0,ANY).GetIma());
  }
  else if (mp.Nrows()==6) A = TOPUP::MovePar2Matrix(mp,sm.Scan(0,ANY).GetIma());
  else throw EddyException("EDDY::PostEddyCFImpl::GetTransformedIma: size of p must be 1 or 6");
  for (unsigned int i=0; i<indx.size(); i++) {
    NEWMAT::Matrix iM = sm.Scan(indx[i],ANY).InverseMovementMatrix();
    iM = A*iM;
    NEWMAT::ColumnVector nmp = TOPUP::Matrix2MovePar(iM.i(),sm.Scan(0,ANY).GetIma());
    sm.Scan(indx[i],ANY).SetParams(nmp,EDDY::MOVEMENT);
  }
  return;
} EddyCatch

void PEASUtils::align_shells_using_MI(// Input
				      const EddyCommandLineOptions&                      clo,
				      bool                                               pe_only,
				      // Input/Output
				      ECScanManager&                                     sm,
				      // Output
				      std::vector<double>&                               grpb,
				      std::vector<NEWMAT::ColumnVector>&                 mov_par,
				      std::vector<std::vector<NEWMAT::ColumnVector> >&   cmov_par,
				      std::vector<NEWMAT::ColumnVector>&                 mp_for_updates) EddyTry
{
  std::vector<std::vector<unsigned int> > dwi_indx;

  // Get mean b0 volume
  NEWIMAGE::volume<float>     b0_mask = sm.Scan(0,ANY).GetIma(); b0_mask=1.0;
  std::vector<unsigned int>   b0_indx = sm.GetB0Indicies();
  NEWIMAGE::volume<float>     b0_mean = PEASUtils::get_mean_scan(clo,sm,b0_indx,b0_mask);
  // NEWIMAGE::write_volume(b0_mean,"mean_b0_volume");

  // Get mean volumes for all shells
  if (clo.VeryVerbose()) cout << "Calculating shell means" << endl;
  std::vector<NEWIMAGE::volume<float> >  dwi_means;
  NEWIMAGE::volume<float>                mask = sm.Scan(0,ANY).GetIma(); mask=1.0;
  if (sm.IsShelled()) { // Get shell means
    dwi_indx = sm.GetShellIndicies(grpb);
    dwi_means.resize(dwi_indx.size());
    NEWIMAGE::volume<float> tmpmask;
    for (unsigned int i=0; i<dwi_indx.size(); i++) {
      tmpmask = 1.0;
      dwi_means[i] = PEASUtils::get_mean_scan(clo,sm,dwi_indx[i],tmpmask);
      mask *= tmpmask;
    }
  }
  else { // Get mean of all dwi's
  }
  mask *= b0_mask; // Mask valid for b0 and dwi
  // NEWIMAGE::write_volume(dwi_means[0],"mean_dwi_volume");
  // NEWIMAGE::write_volume(mask,"peas_mask");

  // Register all volumes to b0
  if (clo.VeryVerbose()) cout << "Registering shell means" << endl;
  mov_par.resize(dwi_means.size());
  NEWIMAGE::volume<float> rima;
  #ifndef COMPILE_GPU
  # pragma omp parallel for shared(mov_par)
  #endif
  for (unsigned int i=0; i<dwi_means.size(); i++) {
    if (pe_only) {
      if (clo.VeryVerbose()) cout << "Registering shell " << grpb[i] << " along PE to b0" << endl;
      unsigned int pe_dir = sm.HasPEinX() ? 0 : 1;
      mov_par[i] = PEASUtils::register_volumes_along_PE(b0_mean,dwi_means[i],mask,pe_dir,rima);
      if (clo.VeryVerbose()) cout << "PE-translation = " << mov_par[i] << endl;
    }
    else {
      if (clo.VeryVerbose()) cout << "Registering shell " << grpb[i] << " to b0" << endl;
      mov_par[i] = PEASUtils::register_volumes(b0_mean,dwi_means[i],mask,rima);
      if (clo.VeryVerbose()) cout << "mov_par = " << mov_par[i] << endl;
    }
  }
  // NEWIMAGE::write_volume(rima,"registered_mean_dwi_volume");


  cmov_par.resize(dwi_means.size());
  for (unsigned int i=0; i<dwi_means.size(); i++) {
    cmov_par[i].resize(dwi_means.size());
    #ifndef COMPILE_GPU
    # pragma omp parallel for shared(cmov_par)
    #endif
    for (unsigned int j=i+1; j<dwi_means.size(); j++) {
      if (pe_only) {
	if (clo.VeryVerbose()) cout << "Registering shell " << j << " along PE to shell " << i << endl; cout.flush();
	if (clo.VeryVerbose()) cout << "Registering shell " << grpb[j] << " along PE to shell " << grpb[i] << endl; cout.flush();
	unsigned int pe_dir = sm.HasPEinX() ? 0 : 1;
	cmov_par[i][j] = PEASUtils::register_volumes_along_PE(dwi_means[i],dwi_means[j],mask,pe_dir,rima);
	if (clo.VeryVerbose()) cout << "PE-translation = " << cmov_par[i][j] << endl;
      }
      else {
	if (clo.VeryVerbose()) cout << "Registering shell " << j << " to shell " << i << endl; cout.flush();
	if (clo.VeryVerbose()) cout << "Registering shell " << grpb[j] << " to shell " << grpb[i] << endl; cout.flush();
	cmov_par[i][j] = PEASUtils::register_volumes(dwi_means[i],dwi_means[j],mask,rima);
	if (clo.VeryVerbose()) cout << "cmov_par = " << cmov_par[i][j] << endl;
      }
    }
  }

  // Collate estimates to use
  mp_for_updates = collate_mov_par_estimates_for_use(mov_par,cmov_par,sm.Scan(0,ANY).GetIma());
} EddyCatch

void PEASUtils::write_between_shell_MI_values(const NEWIMAGE::volume<float>&    ref,
					      const NEWIMAGE::volume<float>&    ima,
					      const NEWIMAGE::volume<float>&    mask,
					      const std::string&                fname,
					      const std::vector<unsigned int>&  n,
					      const std::vector<double>&        first,
					      const std::vector<double>&        last) EddyTry
{
  EddyUtils::SetSplineInterp(ima);
  EddyUtils::SetTrilinearInterp(mask);
  PostEddyCF ecf(ref,ima,mask,256);

  std::ofstream file;
  double pi = 3.14159;
  file.open(fname,ios::out|ios::trunc);
  file << setprecision(8);
  NEWMAT::ColumnVector mp(6);
  for (unsigned int xti=0; xti<n[0]; xti++) {
    mp(1) = (n[0]>1) ? first[0] + xti * ((last[0]-first[0]) / (n[0]-1)) : first[0];
    for (unsigned int yti=0; yti<n[1]; yti++) {
      mp(2) = (n[1]>1) ? first[1] + yti * ((last[1]-first[1]) / (n[1]-1)) : first[1];
      for (unsigned int zti=0; zti<n[2]; zti++) {
	mp(3) = (n[2]>1) ? first[2] + zti * ((last[2]-first[2]) / (n[2]-1)) : first[2];
	for (unsigned int xri=0; xri<n[3]; xri++) {
	  mp(4) = (n[3]>1) ? first[3] + xri * ((last[3]-first[3]) / (n[3]-1)) : first[3];
	  mp(4) = pi * mp(4) / 180.0;
	  for (unsigned int yri=0; yri<n[4]; yri++) {
	    mp(5) = (n[4]>1) ? first[4] + yri * ((last[4]-first[4]) / (n[4]-1)) : first[4];
	    mp(5) = pi * mp(5) / 180.0;
	    for (unsigned int zri=0; zri<n[5]; zri++) {
	      mp(6) = (n[5]>1) ? first[5] + zri * ((last[5]-first[5]) / (n[5]-1)) : first[5];
	      mp(6) = pi * mp(6) / 180.0;
	      file << setw(16) << mp(1) << setw(16) << mp(2) << setw(16) << mp(3);
	      file << setw(16) << 180.0*mp(4)/pi << setw(16) << 180.0*mp(5)/pi << setw(16) << 180.0*mp(6)/pi;
	      file << setw(16) << ecf.cf(mp) << endl;
	    }
	  }
	}
      }
    }
  }
  return;
} EddyCatch

void PEASUtils::align_shells_using_interspersed_B0_scans(// Input
							 const EddyCommandLineOptions&                      clo,
							 // Input/Output
							 ECScanManager&                                     sm,
							 // Output
							 std::vector<double>&                               grpb,
							 std::vector<NEWMAT::ColumnVector>&                 mov_par,
							 std::vector<NEWMAT::ColumnVector>&                 mp_for_updates) EddyTry
{
  std::vector<DiffPara>  dpv = sm.GetDiffParas();
  std::vector<unsigned int> grpi;       // A vector with a group index for each Scan in sm. Group 0 pertains to b0.
  EddyUtils::GetGroups(dpv,grpi,grpb);  // N.B. This version of grpb has not had the b0 removed
  std::vector<unsigned int> b0_indx = sm.GetB0Indicies();

  mov_par.resize(grpb.size()-1);
  for (unsigned int i=0; i<grpb.size()-1; i++) mov_par[i].ReSize(6);

  NEWMAT::ColumnVector delta(grpb.size());                 // Delta values between b0 and shells
  NEWMAT::ColumnVector n(grpb.size());                     // Number of delta values per shell
  for (unsigned int i=0; i<6; i++) { // Loop over movement parameters
    delta=0.0; n=0.0;
    for (unsigned int j=0; j<b0_indx.size(); j++) { // Calculate delta-values between b0s and surrounding dwis
      if (b0_indx[j]==0) {
	if (!sm.IsB0(b0_indx[j]+1)) {
	  NEWMAT::ColumnVector b0mp = sm.Scan(b0_indx[j]).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
	  NEWMAT::ColumnVector mp = sm.Scan(b0_indx[j]+1).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
	  delta(grpi[b0_indx[j]+1]) += mp(i+1)-b0mp(i+1);
	  n(grpi[b0_indx[j]+1]) += 1;
	}
      }
      else if (b0_indx[j]==(sm.NScans()-1)) {
	if (!sm.IsB0(b0_indx[j]-1)) {
	  NEWMAT::ColumnVector b0mp = sm.Scan(b0_indx[j]).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
	  NEWMAT::ColumnVector mp = sm.Scan(b0_indx[j]-1).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
	  delta(grpi[b0_indx[j]-1]) += mp(i+1)-b0mp(i+1);
	  n(grpi[b0_indx[j]-1]) += 1;
	}
      }
      else {
	NEWMAT::ColumnVector b0mp = sm.Scan(b0_indx[j]).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
	if (!sm.IsB0(b0_indx[j]-1)) {
	  NEWMAT::ColumnVector mp = sm.Scan(b0_indx[j]-1).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
	  delta(grpi[b0_indx[j]-1]) += mp(i+1)-b0mp(i+1);
	  n(grpi[b0_indx[j]-1]) += 1;
	}
	if (!sm.IsB0(b0_indx[j]+1)) {
	  NEWMAT::ColumnVector mp = sm.Scan(b0_indx[j]+1).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
	  delta(grpi[b0_indx[j]+1]) += mp(i+1)-b0mp(i+1);
	  n(grpi[b0_indx[j]+1]) += 1;
	}
      }
    }
    for (unsigned int j=0; j<grpb.size()-1; j++) {
      mov_par[j](i+1) = delta(j+1) / float(n(j+1));
    }
  }
  mp_for_updates = mov_par;
} EddyCatch

} // End namespace EDDY
