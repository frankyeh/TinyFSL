/////////////////////////////////////////////////////////////////////
///
/// \file EddyGpuUtils.h
/// \brief Declarations of static class with collection of GPU routines used in the eddy project
///
/// The routines declared here are "bridges" on to the actual GPU
/// routines. The interface to these routines only display classes
/// that are part of the "regular" FSL libraries. Hence this file
/// can be safely included by files that know nothing of the GPU
/// and that are compiled by gcc.
///
/// \author Jesper Andersson
/// \version 1.0b, Nov., 2012.
/// \Copyright (C) 2012 University of Oxford
///
/////////////////////////////////////////////////////////////////////

#ifndef EddyGpuUtils_h
#define EddyGpuUtils_h

#include <cstdlib>
#include <cstddef>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <cuda.h>
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"
#include "DiffusionGP.h"
#include "b0Predictor.h"
#include "ECScanClasses.h"
#include "EddyCommandLineOptions.h"

namespace EDDY {

/////////////////////////////////////////////////////////////////////
///
/// \brief This class contains a set of static methods that implement
/// various utility functions for the eddy project implemented on
/// CUDA GPU.
///
/////////////////////////////////////////////////////////////////////
class EddyGpuUtils
{
public:

  /// Loads prediction maker with images unwarped according to current EC estimates
  static std::shared_ptr<DWIPredictionMaker> LoadPredictionMaker(// Input
								 const EddyCommandLineOptions& clo,
							         ScanType                      st,
								 const ECScanManager&          sm,
								 unsigned int                  iter,
								 float                         fwhm,
								 // Output
								 NEWIMAGE::volume<float>&      mask,
								 // Optional input
								 bool                          use_orig=false);

  ///
  static void MakeScatterBrainPredictions(// Input
					  const EddyCommandLineOptions& clo,
					  const ECScanManager&          sm,
					  const std::vector<double>&    hypar,
					  // Output
					  NEWIMAGE::volume4D<float>&    pred,
					  // Optional input
					  bool                          vwbvrot=false);

  /*
  /// Replaces the scans indicated by rm
  static void UpdatePredictionMaker(// Input
				    const EddyCommandLineOptions&        clo,
				    ScanType                             st,
				    const ECScanManager&                 sm,
				    const ReplacementManager&            rm,
				    const NEWIMAGE::volume<float>&       mask,
				    // Input/Output
				    std::shared_ptr<DWIPredictionMaker>  pmp);
  */

  /// Returns a scan corrected for motion and distortions
  static NEWIMAGE::volume<float> GetUnwarpedScan(// Input
						 const EDDY::ECScan&                               scan,
						 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
						 std::shared_ptr<const NEWIMAGE::volume<float> >   bias,
						 bool                                              use_orig,
						 // Optional output
						 NEWIMAGE::volume<float>                           *omask=NULL);

  /// Returns a scan corrected for motion and distortions, helped by the prediction in pred
  static NEWIMAGE::volume<float> GetUnwarpedScan(// Input
						 const EDDY::ECScan&                               scan,
						 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
						 std::shared_ptr<const NEWIMAGE::volume<float> >   bias,
						 const NEWIMAGE::volume<float>&                    pred,
						 bool                                              use_orig,
						 // Optional output
						 NEWIMAGE::volume<float>                           *omask=NULL);

  /// Returns a scan corrected for motion and distortions. Will override slice-to-vol
  static NEWIMAGE::volume<float> GetVolumetricUnwarpedScan(// Input
							   const EDDY::ECScan&                               scan,
							   std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							   std::shared_ptr<const NEWIMAGE::volume<float> >   bias,
							   bool                                              use_orig,
							   // Optional output
							   NEWIMAGE::volume<float>                           *omask=nullptr,
							   NEWIMAGE::volume4D<float>                         *deriv=nullptr);

  /// Returns a scan corrected for motion (scanner->model(sort of))
  static void GetMotionCorrectedScan(// Input
				     const EDDY::ECScan&       scan,
				     bool                      use_orig,
				     // Output
				     NEWIMAGE::volume<float>&  ovol,
				     // Optional output
				     NEWIMAGE::volume<float>   *omask=NULL);

  /// Returns a scan (in model space) warped into observation space
  static NEWIMAGE::volume<float> TransformModelToScanSpace(const EDDY::ECScan&                               scan,
							   const NEWIMAGE::volume<float>&                    mima,
							   std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							   bool                                              jacmod=true);

  static NEWIMAGE::volume4D<float> DerivativesForModelToScanSpaceTransform(const EDDY::ECScan&                               scan,
									   const NEWIMAGE::volume<float>&                    mima,
									   std::shared_ptr<const NEWIMAGE::volume<float> >   susc);

  static NEWIMAGE::volume4D<float> DirectDerivativesForModelToScanSpaceTransform(const EDDY::ECScan&                               scan,
										 const NEWIMAGE::volume<float>&                    mima,
										 std::shared_ptr<const NEWIMAGE::volume<float> >   susc);

  /// Returns a scan convolved with a Gaussian with fwhm in mm
  static NEWIMAGE::volume<float> Smooth(const NEWIMAGE::volume<float>&  ima,
					float                           fwhm);

  /// Detects outlier-slices
  static DiffStatsVector DetectOutliers(// Input
					const EddyCommandLineOptions&             clo,
					ScanType                                  st,
					const std::shared_ptr<DWIPredictionMaker> pmp,
					const NEWIMAGE::volume<float>&            mask,
					const ECScanManager&                      sm,
					// Input/Output
					ReplacementManager&                       rm);

  /// Replaces outlier-slices
  static void ReplaceOutliers(// Input
			      const EddyCommandLineOptions&             clo,
			      ScanType                                  st,
			      const std::shared_ptr<DWIPredictionMaker> pmp,
			      const NEWIMAGE::volume<float>&            mask,
			      const ReplacementManager&                 rm,
			      bool                                      add_noise,
			      // Input/Output
			      ECScanManager&                            sm);

  /// Performs update of movement and EC parameters for one scan.
  static double MovAndECParamUpdate(// Input
				    const NEWIMAGE::volume<float>&                    pred,
				    std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
				    std::shared_ptr<const NEWIMAGE::volume<float> >   bias,
				    const NEWIMAGE::volume<float>&                    pmask,
				    bool                                              cbs,
				    float                                             fwhm,
				    // Input/output
				    EDDY::ECScan&                                     scan);

  /// Performs update of movement and EC parameters for one scan.
  static double MovAndECParamUpdate(// Input
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
				    EDDY::ECScan&                                     scan);
};

} // End namespace EDDY

#endif // End #ifndef EddyGpuUtils_h
