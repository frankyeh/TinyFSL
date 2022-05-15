/////////////////////////////////////////////////////////////////////
///
/// \file EddyInternalGpuUtils.h
/// \brief Declarations of static class with collection of GPU routines used in the eddy project
///
/// \author Jesper Andersson
/// \version 1.0b, Nov., 2012.
/// \Copyright (C) 2012 University of Oxford 
///
/////////////////////////////////////////////////////////////////////

#ifndef EddyInternalGpuUtils_h
#define EddyInternalGpuUtils_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#pragma push
#pragma diag_suppress = code_is_unreachable // Supress warnings from armawrap
#include "CudaVolume.h"
#pragma pop
#include "DiffusionGP.h"
#include "b0Predictor.h"
#include "ECScanClasses.h"
#include "EddyCommandLineOptions.h"

namespace EDDY {

class EddyGpuUtils;         // Forward declaration of friend
class DerivativeCalculator; // Forward declaration of friend

/////////////////////////////////////////////////////////////////////
///
/// \brief This class contains a set of static methods that implement
/// various utility functions for the eddy project implemented on 
/// CUDA GPU.
///
/////////////////////////////////////////////////////////////////////
class EddyInternalGpuUtils
{
private:
  friend class DerivativeCalculator;
  friend class EddyGpuUtils;
  friend class PostEddyCFImpl;

  static const int threads_per_block_make_deriv = 128;
  static const int threads_per_block_ec_field = 128;

  /// Loads predicition maker with scans unwarped according to current EC estimates
  static void load_prediction_maker(// Input
				    const EddyCommandLineOptions&        clo,
				    ScanType                             st,
				    const ECScanManager&                 sm,
				    unsigned int                         iter,
				    float                                fwhm,
		                    bool                                 use_orig,
				    // Output
				    std::shared_ptr<DWIPredictionMaker>  pmp,
				    NEWIMAGE::volume<float>&             mask);

  /*
  static void update_prediction_maker(// Input
				      const EddyCommandLineOptions&          clo,
				      ScanType                               st,
				      const ECScanManager&                   sm,
				      const ReplacementManager&              rm,
				      const NEWIMAGE::volume<float>&         mask,
				      // Input/Output
				      std::shared_ptr<DWIPredictionMaker>    pmp);
  */

  /// Motion corrects a scan from native (scanner) space to model space
  static void get_motion_corrected_scan(// Input
					const EDDY::ECScan&     scan,
					bool                    use_orig,
					// Output
					EDDY::CudaVolume&       oima,
					// Optional output
					EDDY::CudaVolume&       omask);

  /// Unwarps a scan from native (scanner) space to model space
  static void get_unwarped_scan(// Input
				const EDDY::ECScan&        scan,
				const EDDY::CudaVolume&    susc,
				const EDDY::CudaVolume&    bias,
				const EDDY::CudaVolume&    pred,
				bool                       jacmod,
				bool                       use_orig,
				// Output
				EDDY::CudaVolume&          oima,
				EDDY::CudaVolume&          omask);

  /// Unwarps a scan from native (scanner) space to model space. It uses a 
  /// volumetric model regardless of what the movement degrees of freedom
  /// is set to.
  static void get_volumetric_unwarped_scan(// Input
					   const EDDY::ECScan&        scan,
					   const EDDY::CudaVolume&    susc,
					   const EDDY::CudaVolume&    bias,
					   bool                       jacmod,
					   bool                       use_orig,
					   // Output
					   EDDY::CudaVolume&          oima,
					   EDDY::CudaVolume&          omask,
					   EDDY::CudaVolume4D&        deriv);

  static void detect_outliers(// Input
			      const EddyCommandLineOptions&             clo,
			      ScanType                                  st,
			      const std::shared_ptr<DWIPredictionMaker> pmp,
			      const NEWIMAGE::volume<float>&            mask,
			      const ECScanManager&                      sm,
			      // Input/Output
			      ReplacementManager&                       rm,
			      DiffStatsVector&                          dsv);

  static void replace_outliers(// Input
			       const EddyCommandLineOptions&             clo,
			       ScanType                                  st,
			       const std::shared_ptr<DWIPredictionMaker> pmp,
			       const NEWIMAGE::volume<float>&            mask,
			       const ReplacementManager&                 rm,
			       bool                                      add_noise,
			       // Input/Output
			       ECScanManager&                            sm);

  /// Calculates the total field for transform scanner->model
  static void field_for_scan_to_model_transform(// Input
						const EDDY::ECScan&     scan,
						const EDDY::CudaVolume& susc,
						// Output
						EDDY::CudaVolume4D&     dfield,
						EDDY::CudaVolume&       omask,
						EDDY::CudaVolume&       jac);

  /// Calculates the total field for transform scanner->model, forcing volumetric field regardless of slice-to-vol
  static void field_for_scan_to_model_volumetric_transform(// Input
							   const EDDY::ECScan&            scan,
							   const EDDY::CudaVolume&        susc,
							   // Output
							   EDDY::CudaVolume4D&            dfield,
							   // Optional output
							   EDDY::CudaVolume&              omask,
							   EDDY::CudaVolume&              jac);

  /// Performs update of parameters given by whichp for one scan.
  static double param_update(// Input
			     const NEWIMAGE::volume<float>&                  pred,     // Prediction in model space
			     std::shared_ptr<const NEWIMAGE::volume<float> > susc,     // Susc-induced off-resonance field
			     std::shared_ptr<const NEWIMAGE::volume<float> > bias,     // Recieve bias field			  
   			     const NEWIMAGE::volume<float>&                  pmask, 
			     EDDY::Parameters                                whichp,   // Which parameters to update    
			     bool                                            cbs,      // Check (success of parameters) Before Set
			     float                                           fwhm,
			     // These inputs are for debug purposes only
			     unsigned int                                    scindex,
			     unsigned int                                    iter,
			     unsigned int                                    level,
			     // Input/output
			     EDDY::ECScan&                                   scan,     // Scan we want to register to pred
			     // Optional output
			     NEWMAT::ColumnVector                            *rupdate);// Vector of updates


  /// The writing of debug info was lifted out of param_update to simplify the code
  static void write_debug_info_for_param_update(const EDDY::ECScan&               scan,							  
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
						const NEWMAT::ColumnVector&       update);

  /// Transforms a volume in model space to scan/observation space, leaving the result on the gpu
  static void transform_model_to_scan_space(// Input
					    const EDDY::CudaVolume&     pred,
					    const EDDY::ECScan&         scan,
					    const EDDY::CudaVolume&     susc,
					    bool                        jacmod,
					    // Output
					    EDDY::CudaVolume&           oima,
					    EDDY::CudaVolume&           omask,
					    // Optional output
					    EDDY::CudaVolume&           jac,
					    EDDY::CudaVolume4D&         grad);

  /// Calculates field used for transforming model to scan space
  static void field_for_model_to_scan_transform(// Input
						const EDDY::ECScan&       scan,
						const EDDY::CudaVolume&   susc,
						// Output
						EDDY::CudaVolume4D&       dfield,
						EDDY::CudaVolume&         omask,
						// Optional output
						EDDY::CudaVolume&         jac);

  /// Transform regular-grid coordinates in model space to scan space
  static EDDY::CudaImageCoordinates transform_coordinates_from_model_to_scan_space(// Input
										   const EDDY::CudaVolume&     pred,
										   const EDDY::ECScan&         scan,
										   const EDDY::CudaVolume&     susc,
										   // Output
										   EDDY::CudaImageCoordinates& coord,
										   // Optional Output
										   EDDY::CudaVolume&           omask,
										   EDDY::CudaVolume&           jac);

  /// Calculates partiald derivatives with respect EC and movement parameters
  static void get_partial_derivatives_in_scan_space(const EDDY::CudaVolume&  pred,
						    const EDDY::ECScan&      scan,
						    const EDDY::CudaVolume&  susc,
						    EDDY::Parameters         whichp,
						    EDDY::CudaVolume4D&      derivs);

  static void get_direct_partial_derivatives_in_scan_space(const EDDY::CudaVolume& pred,
							   const EDDY::ECScan&     scan,
							   const EDDY::CudaVolume& susc,
							   EDDY::Parameters        whichp,
							   EDDY::CudaVolume4D&     derivs);

  static void make_deriv_from_components(const EDDY::CudaImageCoordinates&  coord,
					 const EDDY::CudaVolume4D&          grad,
					 const EDDY::CudaVolume&            base,
					 const EDDY::CudaVolume&            jac,
					 const EDDY::CudaVolume&            basejac,
					 float                              dstep,
					 EDDY::CudaVolume4D&                deriv,
					 unsigned int                       indx);

  static NEWMAT::Matrix make_XtX(const EDDY::CudaVolume4D&  X,
				 const EDDY::CudaVolume&    mask);

  static NEWMAT::Matrix make_XtX_cuBLAS(const EDDY::CudaVolume4D&  X);

  static NEWMAT::ColumnVector make_Xty(const EDDY::CudaVolume4D&  X,
				       const EDDY::CudaVolume&    y,
				       const EDDY::CudaVolume&    mask);

  static void make_scatter_brain_predictions(// Input
					     const EddyCommandLineOptions&  clo,
					     const ECScanManager&           sm,
					     const std::vector<double>&     hypar,
					     // Output
					     NEWIMAGE::volume4D<float>&     pred,
					     // Optional input
					     bool                           vwbvrot=false);

  static void general_transform(const EDDY::CudaVolume&    inima,
				const NEWMAT::Matrix&      A,
				const EDDY::CudaVolume4D&  dfield,
				const NEWMAT::Matrix&      M,
				EDDY::CudaVolume&          oima,
				EDDY::CudaVolume4D&        deriv,
				EDDY::CudaVolume&          omask);

  static void general_transform(// Input
				const EDDY::CudaVolume&             inima,
				const std::vector<NEWMAT::Matrix>&  A,
				const EDDY::CudaVolume4D&           dfield,
				const std::vector<NEWMAT::Matrix>&  M,
				// Output
				EDDY::CudaVolume&                   oima,
				// Optional output
				EDDY::CudaVolume4D&                 deriv,
				EDDY::CudaVolume&                   omask);

  static void general_slice_to_vol_transform(// Input
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
					     EDDY::CudaVolume&                   omask);

  static void half_way_slice_to_vol_transform(// Input
					      const EDDY::CudaVolume&             inima,
					      const std::vector<NEWMAT::Matrix>&  A,
					      const EDDY::CudaVolume4D&           dfield,
					      const EDDY::CudaVolume&             jac,
					      // Output
					      EDDY::CudaVolume&                   hwima,
					      EDDY::CudaVolume&                   zcoordV,
					      // Optional output
					      EDDY::CudaVolume&                   oima,
					      EDDY::CudaVolume&                   omask);

  static void affine_transform(const EDDY::CudaVolume&    inima,
			       const NEWMAT::Matrix&      R,
			       EDDY::CudaVolume&          oima,
			       EDDY::CudaVolume4D&        deriv,
			       EDDY::CudaVolume&          omask);

  static void affine_transform(const EDDY::CudaVolume&             inima,
			       const std::vector<NEWMAT::Matrix>&  R,
			       EDDY::CudaVolume&                   oima,
			       EDDY::CudaVolume4D&                 deriv,
			       EDDY::CudaVolume&                   omask);

  /// Caclulates EC field given current parameters
  static void get_ec_field(// Input
			   const EDDY::ECScan&        scan,
			   // Output
			   EDDY::CudaVolume&          ecfield);

};

/////////////////////////////////////////////////////////////////////
///
/// \brief This class contains a set of static methods that implement
/// various field related utility functions for the eddy project 
/// implemented on CUDA GPU.
///
/////////////////////////////////////////////////////////////////////
class FieldGpuUtils
{
private:
  friend class EddyInternalGpuUtils;
  friend class DerivativeCalculator;

  static const int threads_per_block_invert_field = 128;

  static void Hz2VoxelDisplacements(const EDDY::CudaVolume&  hzfield,
				    const EDDY::AcqPara&     acqp,
				    EDDY::CudaVolume4D&      dfield);

  static void Voxel2MMDisplacements(EDDY::CudaVolume4D&      dfield);

  static void InvertDisplacementField(// Input
				      const EDDY::CudaVolume4D&  dfield,
				      const EDDY::AcqPara&       acqp,
				      const EDDY::CudaVolume&    inmask,
				      // Output
				      EDDY::CudaVolume4D&        idfield,
				      EDDY::CudaVolume&          omask);
    
  static void GetJacobian(// Input
			  const EDDY::CudaVolume4D&  dfield,
			  const EDDY::AcqPara&       acqp,
			  // Output
			  EDDY::CudaVolume&          jac);

  static void GetDiscreteJacobian(// Input
				  const EDDY::CudaVolume4D&  dfield,
				  const EDDY::AcqPara&       acqp,
				  // Output
				  EDDY::CudaVolume&          jac);

};

} // End namespace EDDY

#endif // End #ifndef EddyInternalGpuUtils_h
