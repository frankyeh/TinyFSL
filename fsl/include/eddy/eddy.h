/////////////////////////////////////////////////////////////////////
///
/// \file eddy.h
/// \brief Contains declarations of some very high level functions for eddy
///
/// This file contains declarations for some very high level functions
/// that are called in eddy.cpp.
///
/// \author Jesper Andersson
/// \version 1.0b, Nov., 2012.
/// \Copyright (C) 2012 University of Oxford
///
/////////////////////////////////////////////////////////////////////

#ifndef eddy_h
#define eddy_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include "armawrap/newmat.h"
#include "EddyHelperClasses.h"
#include "ECScanClasses.h"
#include "DiffusionGP.h"
#include "b0Predictor.h"
#include "EddyUtils.h"
#include "EddyCommandLineOptions.h"

namespace EDDY {

/// A very high-level global function that registers all the scans (b0 and dwis) in sm using a volume-to-volume model.
ReplacementManager *DoVolumeToVolumeRegistration(// Input
						 const EddyCommandLineOptions&  clo,
						 // Input/Output
						 ECScanManager&                 sm);

/// A very high-level global function that registers all the scans (b0 and dwis) in sm using a slice-to-volume model.
ReplacementManager *DoSliceToVolumeRegistration(// Input
						const EddyCommandLineOptions&  clo,
						unsigned int                   oi,        // Order index
						bool                           dol,       // Detect outliers?
						// Input/Output
						ECScanManager&                 sm,
						ReplacementManager             *dwi_rm);

/// A very high-level global function that estimates a bias field
void EstimateBiasField(// Input
		       const EddyCommandLineOptions&  clo,
		       double                         ksp,
		       double                         lambda,
		       // Input/output
		       ECScanManager&                 sm);

/// Global function that registers a set of scans together
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
			     NEWMAT::Matrix&                phist);

/// Global function that performs final check for outliers without error-variance fudging.
ReplacementManager *FinalOLCheck(// Input
				 const EddyCommandLineOptions&  clo,
				 // Input/output
				 ReplacementManager             *rm,
				 ECScanManager&                 sm);

/// Global function that detect outlier slices and replaces them by their expectation
DiffStatsVector DetectAndReplaceOutliers(// Input
					 const EddyCommandLineOptions& clo,
					 ScanType                      st,
					 // Input/Output
					 ECScanManager&                sm,
					 ReplacementManager&           rm);

/// Global function that Loads up the prediction maker with unwarped scans
std::shared_ptr<DWIPredictionMaker> LoadPredictionMaker(// Input
							const EddyCommandLineOptions& clo,
							ScanType                      st,
							const ECScanManager&          sm,
							unsigned int                  iter,
							float                         fwhm,
							// Output
							NEWIMAGE::volume<float>&      mask,
							// Optional input
							bool                          use_orig=false);

/*
/// Global function that replaces selected (by rm) volumes in prediction maker
void UpdatePredictionMaker(// Input
			   const EddyCommandLineOptions&          clo,
			   ScanType                               st,
			   const ECScanManager&                   sm,
			   const ReplacementManager&              rm,
			   const NEWIMAGE::volume<float>&         mask,
			   // Input/Output
			   std::shared_ptr<DWIPredictionMaker>    pmp);
*/

/// Looks for outlier slices
DiffStatsVector DetectOutliers(// Input
			       const EddyCommandLineOptions&               clo,
			       ScanType                                    st,
			       const std::shared_ptr<DWIPredictionMaker>   pmp,
			       const NEWIMAGE::volume<float>&              mask,
			       const ECScanManager&                        sm,
			       // Input/Output
			       ReplacementManager&                         rm);

/// Replaces outlier slices with their predictions
void ReplaceOutliers(// Input
		     const EddyCommandLineOptions&               clo,
		     ScanType                                    st,
		     const std::shared_ptr<DWIPredictionMaker>   pmp,
		     const NEWIMAGE::volume<float>&              mask,
		     const ReplacementManager&                   rm,
		     bool                                        add_noise,
		     // Input/Output
		     ECScanManager&                              sm);

/// Get predictions to help with slice-to-vol resampling
std::vector<double> GetPredictionsForResampling(// Input
						const EddyCommandLineOptions&    clo,
						ScanType                         st,
						const ECScanManager&             sm,
						// Output
						NEWIMAGE::volume4D<float>&       pred);

void GetScatterBrainPredictions(// Input
                                const EddyCommandLineOptions&    clo,
				ScanType                         st,
				ECScanManager&                   sm,
				const std::vector<double>&       hypar,
				// Output
				NEWIMAGE::volume4D<float>&       pred,
				// Optional input
				bool                             vwbvrot=false);

/// Calculate maps of CNR and SNR
void CalculateCNRMaps(// Input
		      const EddyCommandLineOptions&               clo,
		      const ECScanManager&                        sm,
		      // Output
		      std::shared_ptr<NEWIMAGE::volume4D<float> > std_cnr,
		      std::shared_ptr<NEWIMAGE::volume4D<float> > range_cnr,
                      std::shared_ptr<NEWIMAGE::volume<float> >   b0_snr,
		      std::shared_ptr<NEWIMAGE::volume4D<float> > residuals);

/// Write maps of CNR and SNR
void WriteCNRMaps(// Input
		  const EddyCommandLineOptions&   clo,
		  const ECScanManager&            sm,
		  const std::string&              spatial_fname,
		  const std::string&              range_fname,
		  const std::string&              temporal_fname);

/// Global function that Generates diagnostic information for subsequent analysis
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
		 NEWMAT::Matrix&                phist);

/// Global function that adds rotation to all volumes.
void AddRotation(ECScanManager&               sm,
		 const NEWMAT::ColumnVector&  rp);

/// Global function that prints out (for debugging) MI values between shells
void PrintMIValues(const EddyCommandLineOptions&  clo,
                   const ECScanManager&           sm,
                   const std::string&             fname,
                   bool                           write_planes);

} // End namespace EDDY


#endif // End #ifndef eddy_h
